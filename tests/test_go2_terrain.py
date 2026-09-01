"""Regression checks for physical Go2 terrain locomotion/landing tasks."""

from __future__ import annotations

import unittest

import mujoco
import numpy as np

from landing_rl.go2_legged_loco_environment import Go2LeggedLocoEnv
from landing_rl.go2_onnx_inference import terrain_hud_label
from landing_rl.go2_qr_environment import (
    DRONE_OBSERVATION_NAMES,
    Go2BackQrLandingEnv,
    build_go2_landing_xml,
    configure_go2_sole_only_ground_contact,
)
from landing_rl.go2_terrain import (
    ROUGH_LEVEL_AMPLITUDE_M,
    ROUGH_HFIELD_NAME,
    GRAVEL_HEIGHT_AMPLITUDE_M,
    GRAVEL_HFIELD_NAME,
    GRAVEL_LENGTH_M,
    GRAVEL_SLOPE_GRADE,
    SLOPE_GRADE,
    SLOPE_GRADE_PERCENT,
    configure_rough_terrain,
    gravel_rock_specs,
    terrain_course_bounds,
    terrain_edge_clearance_m,
    terrain_geom_names,
    terrain_metadata,
    terrain_height_at,
)


class Go2TerrainTest(unittest.TestCase):
    def test_finite_course_bounds_and_signed_edge_clearance(self) -> None:
        self.assertEqual(terrain_course_bounds("flat"), None)
        self.assertTrue(np.isinf(terrain_edge_clearance_m("flat", 1_000.0, 1_000.0)))
        self.assertGreater(terrain_edge_clearance_m("slope_up", 0.0, 0.0), 0.0)
        self.assertLess(terrain_edge_clearance_m("slope_down", 15.1, 0.0), 0.0)
        self.assertGreater(terrain_edge_clearance_m("rough", 1.0, 0.0), 0.0)
        self.assertLess(terrain_edge_clearance_m("rough", 1.0, 1.25), 0.0)

    def test_slope_is_a_visible_physical_collision_surface(self) -> None:
        model = mujoco.MjModel.from_xml_string(build_go2_landing_xml(include_drone=False, terrain_task="slope_up"))
        slope = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "terrain_slope_up")
        self.assertGreaterEqual(slope, 0)
        self.assertGreater(model.geom_contype[slope], 0)
        self.assertGreater(model.geom_conaffinity[slope], 0)
        self.assertAlmostEqual(
            terrain_height_at("slope_up", 14.0, 0.0) - terrain_height_at("slope_up", 0.0, 0.0),
            14.0 * SLOPE_GRADE,
            places=6,
        )
        self.assertAlmostEqual(
            terrain_height_at("slope_down", 0.0, 0.0) - terrain_height_at("slope_down", 14.0, 0.0),
            14.0 * SLOPE_GRADE,
            places=6,
        )
        self.assertEqual(terrain_metadata("slope_up")["slope_grade_percent"], SLOPE_GRADE_PERCENT)

    def test_rough_level_updates_real_collision_hfield(self) -> None:
        model = mujoco.MjModel.from_xml_string(build_go2_landing_xml(include_drone=False, terrain_task="rough"))
        terrain = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "terrain_rough")
        hfield = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, ROUGH_HFIELD_NAME)
        self.assertGreaterEqual(terrain, 0)
        self.assertGreaterEqual(hfield, 0)
        self.assertGreater(model.geom_contype[terrain], 0)
        count = int(model.hfield_nrow[hfield] * model.hfield_ncol[hfield])
        start = int(model.hfield_adr[hfield])
        configure_rough_terrain(model, level=3)
        level3 = model.hfield_data[start:start + count].copy()
        configure_rough_terrain(model, level=1)
        level1 = model.hfield_data[start:start + count].copy()
        self.assertGreater(float(np.max(np.abs(level3 - level1))), 0.01)
        self.assertGreater(
            abs(terrain_height_at("rough", 1.2, 0.2, rough_level=3) - terrain_height_at("rough", 1.2, 0.2, rough_level=1)),
            0.001,
        )

    def test_rough_level_three_has_visible_foot_scale_relief(self) -> None:
        x_samples = np.linspace(-0.35, 14.85, 81)
        y_samples = np.linspace(-1.05, 1.05, 17)
        heights = np.array(
            [
                terrain_height_at("rough", float(x), float(y), rough_level=3)
                for y in y_samples
                for x in x_samples
            ],
            dtype=np.float64,
        ).reshape(len(y_samples), len(x_samples))
        # Preserve the advertised level-3 relief without reverting to vertical
        # box walls: at least 70% of the 160 mm nominal peak-to-peak range is
        # represented by the continuous collision surface.
        self.assertGreater(float(heights.max() - heights.min()), 1.4 * ROUGH_LEVEL_AMPLITUDE_M[3])
        # A non-zero local height difference over a 0.19 m sampling interval
        # catches regressions back to a single long, visually-flat swell.
        self.assertGreater(float(np.percentile(np.abs(np.diff(heights, axis=1)), 90)), 0.002)

    def test_gravel_is_a_long_course_with_individual_collision_stones(self) -> None:
        model = mujoco.MjModel.from_xml_string(build_go2_landing_xml(include_drone=False, terrain_task="gravel"))
        terrain = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "terrain_gravel")
        hfield = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, GRAVEL_HFIELD_NAME)
        self.assertGreaterEqual(terrain, 0)
        self.assertGreaterEqual(hfield, 0)
        self.assertGreater(model.geom_contype[terrain], 0)
        self.assertGreater(GRAVEL_LENGTH_M, 30.0)
        rocks = gravel_rock_specs()
        self.assertGreaterEqual(len(rocks), 1_500)
        self.assertEqual(len(terrain_geom_names("gravel")), len(rocks) + 1)
        first_band, first_index, *_ = rocks[0]
        first_rock = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, f"terrain_gravel_rock_{first_band}_{first_index}"
        )
        self.assertGreaterEqual(first_rock, 0)
        self.assertGreater(model.geom_contype[first_rock], 0)
        x_samples = np.linspace(-0.6, 25.0, 241)
        heights = np.array([terrain_height_at("gravel", float(x), 0.0) for x in x_samples])
        # ``terrain_height_at`` deliberately returns the continuous soil base
        # (for reset stability), not an arbitrary clast top.  It must still
        # carry the declared gentle grade and sub-centimetre undulation.
        self.assertGreater(float(heights.max() - heights.min()), 0.60 * GRAVEL_HEIGHT_AMPLITUDE_M)
        self.assertAlmostEqual(
            terrain_height_at("gravel", 25.0, 0.0) - terrain_height_at("gravel", 0.0, 0.0),
            25.0 * GRAVEL_SLOPE_GRADE,
            delta=0.025,
        )
        # The soil remains continuous between individual embedded stones: no
        # vertical wall is allowed between adjacent 10.7 cm samples.
        self.assertLess(float(np.max(np.abs(np.diff(heights)))), 0.050)
        self.assertGreater(terrain_edge_clearance_m("gravel", 10.0, 0.0), 0.0)
        self.assertLess(terrain_edge_clearance_m("gravel", 34.0, 0.0), 0.0)

    def test_rough_locomotion_reset_uses_selected_level(self) -> None:
        env = Go2LeggedLocoEnv(
            terrain_task="rough", rough_level=3, domain_randomization=False, sensor_noise=False, max_steps=4
        )
        observation, info = env.reset(seed=19)
        self.assertEqual(observation.shape, (450,))
        self.assertEqual(info["rough_level"], 3.0)
        _, _, _, _, step_info = env.step(np.zeros(12, dtype=np.float32))
        self.assertEqual(step_info["terrain_rough_level"], 3.0)
        self.assertEqual(step_info["root_wrench_max_abs"], 0.0)
        env.close()

    def test_landing_scene_exposes_terrain_only_as_offline_diagnostic(self) -> None:
        env = Go2BackQrLandingEnv(terrain_task="slope_up", difficulty="easy")
        observation, _ = env.reset(seed=23)
        self.assertEqual(observation.shape, (7,))
        self.assertFalse(any("terrain" in name or "go2" in name for name in DRONE_OBSERVATION_NAMES))
        _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
        self.assertIn("terrain_ground_height_m", info)
        self.assertIn("offline_sim_terrain_course_inside", info)
        self.assertGreaterEqual(info["offline_sim_terrain_boundary_clearance_m"], 0.0)
        self.assertEqual(info["go2_assist_force_n"], 0.0)
        env.close()

    def test_terrain_exit_latches_and_blocks_landing_success(self) -> None:
        env = Go2BackQrLandingEnv(terrain_task="slope_up", difficulty="easy")
        env.reset(seed=23)
        # Put the actual Go2 free body beyond the 16 m collision course.  The
        # QR deck follows it rigidly, so this is a direct check that terrain
        # departure cannot later be reported as a successful X500 landing.
        env.data.qpos[0] = 16.0
        mujoco.mj_forward(env.model, env.data)
        self.assertFalse(env._terrain_course_status()[0])
        _, _, terminated, _, info = env.step(np.zeros(2, dtype=np.float32))
        self.assertTrue(terminated)
        self.assertEqual(info["success"], 0.0)
        self.assertEqual(info["offline_sim_terrain_course_breach"], 1.0)
        env.close()

    def test_gravel_requires_a_moving_go2_for_landing(self) -> None:
        env = Go2BackQrLandingEnv(terrain_task="gravel", difficulty="medium")
        observation, info = env.reset(seed=31)
        self.assertEqual(observation.shape, (7,))
        self.assertEqual(info["terrain_task"], "gravel")
        self.assertAlmostEqual(float(env._path_command(0.0)[0]), 0.75, places=6)
        # Simulate the explicit replay gate, not a fake contact: a stopped
        # deck must terminate without ever receiving a landing success label.
        env._go2_motion_violation = True
        _, _, terminated, _, step_info = env.step(np.zeros(2, dtype=np.float32))
        self.assertTrue(terminated)
        self.assertEqual(step_info["success"], 0.0)
        self.assertEqual(step_info["offline_sim_go2_motion_violation"], 1.0)
        env.close()

    def test_gravel_uses_only_official_rubber_paw_ground_contacts(self) -> None:
        model = mujoco.MjModel.from_xml_string(build_go2_landing_xml(include_drone=False, terrain_task="gravel"))
        soles, nonsole = configure_go2_sole_only_ground_contact(model)
        self.assertEqual(len(soles), 4)
        self.assertGreater(len(nonsole), 0)
        self.assertTrue(np.all(model.geom_contype[soles] == 1))
        self.assertTrue(np.all(model.geom_conaffinity[soles] == 1))
        self.assertTrue(np.all(model.geom_contype[nonsole] == 0))
        self.assertTrue(np.all(model.geom_conaffinity[nonsole] == 0))

        env = Go2LeggedLocoEnv(
            terrain_task="gravel", domain_randomization=False, sensor_noise=False, max_steps=120
        )
        env.reset(seed=20262001)
        env._command[:] = (0.75, 0.0, 0.0)
        env._command_change_at = 1_000_000
        maximum_sole_force = 0.0
        for _ in range(100):
            _, _, terminated, truncated, info = env.step(np.zeros(12, dtype=np.float32))
            maximum_sole_force = max(maximum_sole_force, info["sole_normal_force_n"])
            self.assertEqual(info["nonsole_terrain_contacts"], 0.0)
            self.assertEqual(info["nonsole_terrain_violation"], 0.0)
            self.assertFalse(terminated)
            self.assertFalse(truncated)
        self.assertGreater(maximum_sole_force, 1.0)
        env.close()

    def test_video_hud_label_is_font_safe_ascii(self) -> None:
        self.assertEqual(terrain_hud_label("slope_up", None), "UPHILL 10pct")
        self.assertEqual(terrain_hud_label("rough", 3), "ROUGH LEVEL 3")
        self.assertEqual(terrain_hud_label("gravel", None), "GRAVEL ROAD")


if __name__ == "__main__":
    unittest.main()
