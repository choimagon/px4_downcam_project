"""Regression checks for physical Go2 terrain locomotion/landing tasks."""

from __future__ import annotations

import unittest

import mujoco
import numpy as np

from landing_rl.go2_legged_loco_environment import Go2LeggedLocoEnv
from landing_rl.go2_onnx_inference import terrain_hud_label
from landing_rl.go2_qr_environment import DRONE_OBSERVATION_NAMES, Go2BackQrLandingEnv, build_go2_landing_xml
from landing_rl.go2_terrain import (
    ROUGH_HFIELD_NAME,
    SLOPE_GRADE,
    SLOPE_GRADE_PERCENT,
    configure_rough_terrain,
    terrain_metadata,
    terrain_height_at,
)


class Go2TerrainTest(unittest.TestCase):
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
        self.assertEqual(info["go2_assist_force_n"], 0.0)
        env.close()

    def test_video_hud_label_is_font_safe_ascii(self) -> None:
        self.assertEqual(terrain_hud_label("slope_up", None), "UPHILL 10pct")
        self.assertEqual(terrain_hud_label("rough", 3), "ROUGH LEVEL 3")


if __name__ == "__main__":
    unittest.main()
