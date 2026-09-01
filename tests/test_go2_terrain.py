"""Regression checks for physical Go2 terrain locomotion/landing tasks."""

from __future__ import annotations

import unittest

import mujoco
import numpy as np

from landing_rl.go2_legged_loco_environment import Go2LeggedLocoEnv
from landing_rl.go2_onnx_inference import terrain_hud_label
from landing_rl.go2_qr_environment import DRONE_OBSERVATION_NAMES, Go2BackQrLandingEnv, build_go2_landing_xml
from landing_rl.go2_terrain import (
    ROUGH_LEVEL_AMPLITUDE_M,
    ROUGH_HFIELD_NAME,
    SLOPE_GRADE,
    SLOPE_GRADE_PERCENT,
    configure_rough_terrain,
    terrain_course_bounds,
    terrain_edge_clearance_m,
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

    def test_video_hud_label_is_font_safe_ascii(self) -> None:
        self.assertEqual(terrain_hud_label("slope_up", None), "UPHILL 10pct")
        self.assertEqual(terrain_hud_label("rough", 3), "ROUGH LEVEL 3")


if __name__ == "__main__":
    unittest.main()
