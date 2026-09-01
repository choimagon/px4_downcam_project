from __future__ import annotations

import unittest
from pathlib import Path

import cv2
import numpy as np

from landing_rl.environment import QrPrecisionLandingEnv
from landing_rl.scenario import (
    INNER_RING_RADIUS_M,
    OUTER_RING_RADIUS_M,
    QR_SIZE_M,
    random_evaluation_motion_profile,
    random_training_motion_profile,
    random_wavy_motion_profile,
    sample_annulus_start,
)
from landing_rl.vision import QrDetector


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class QrLandingTests(unittest.TestCase):
    def test_generated_gazebo_texture_decodes_to_expected_payload(self) -> None:
        texture = cv2.imread(
            str(PROJECT_ROOT / "PX4-Autopilot/Tools/simulation/gz/models/qr_landing_pad/qr_landing_pad.png"),
            cv2.IMREAD_COLOR,
        )
        self.assertIsNotNone(texture)
        detection = QrDetector().detect(texture)
        self.assertIsNotNone(detection)
        assert detection is not None
        self.assertEqual(detection.payload, "QR")
        self.assertAlmostEqual(detection.normalized_error[0], 0.0, delta=0.01)
        self.assertAlmostEqual(detection.normalized_error[1], 0.0, delta=0.01)

    def test_environment_exposes_camera_compatible_observation_and_lateral_action(self) -> None:
        environment = QrPrecisionLandingEnv(seed=12)
        observation, _ = environment.reset(seed=12)
        self.assertEqual(observation.shape, (6,))
        self.assertEqual(environment.action_space.shape, (2,))
        next_observation, reward, _, _, info = environment.step(np.zeros(2, dtype=np.float32))
        self.assertTrue(environment.observation_space.contains(next_observation))
        self.assertIsInstance(reward, float)
        self.assertIn("aligned_streak", info)

    def test_annulus_start_is_seeded_and_stays_between_the_visible_rings(self) -> None:
        first = sample_annulus_start(77)
        second = sample_annulus_start(77)
        self.assertEqual(first, second)
        self.assertGreater(first.radius_m, INNER_RING_RADIUS_M)
        self.assertLess(first.radius_m, OUTER_RING_RADIUS_M)
        self.assertAlmostEqual(float(np.hypot(first.x_m, first.y_m)), first.radius_m, places=6)

    def test_gazebo_world_uses_half_size_qr_and_annulus_rings(self) -> None:
        qr_model = (
            PROJECT_ROOT / "PX4-Autopilot/Tools/simulation/gz/models/qr_landing_pad/model.sdf"
        ).read_text(encoding="utf-8")
        world = (PROJECT_ROOT / "PX4-Autopilot/Tools/simulation/gz/worlds/aruco.sdf").read_text(encoding="utf-8-sig")
        rings = (
            PROJECT_ROOT / "PX4-Autopilot/Tools/simulation/gz/models/landing_zone_rings/model.sdf"
        ).read_text(encoding="utf-8")
        self.assertIn(f"<size>{QR_SIZE_M:.1f} {QR_SIZE_M:.1f}</size>", qr_model)
        self.assertIn("model://landing_zone_rings", world)
        self.assertEqual(rings.count('name="inner_2m_'), 32)
        self.assertEqual(rings.count('name="outer_7m_'), 32)

    def test_seeded_wavy_profile_is_smooth_bounded_and_reproducible(self) -> None:
        first = random_wavy_motion_profile(913)
        second = random_wavy_motion_profile(913)
        self.assertEqual(first, second)
        sampled_speeds = [float(np.hypot(*first.velocity_at(time_s))) for time_s in np.linspace(0.0, 80.0, 401)]
        self.assertGreater(min(sampled_speeds), 0.15)
        self.assertLess(max(sampled_speeds), 0.39)
        velocity_changes = [
            float(np.hypot(*(np.subtract(first.velocity_at(after), first.velocity_at(before)))))
            for before, after in zip(np.linspace(0.0, 20.0, 201), np.linspace(0.1, 20.1, 201))
        ]
        self.assertLess(max(velocity_changes), 0.03)
        roll, pitch, yaw, heave = first.wave_at(7.0)
        self.assertLessEqual(abs(roll), first.wave_roll_deg)
        self.assertLessEqual(abs(pitch), first.wave_pitch_deg)
        self.assertLessEqual(abs(yaw), first.wave_yaw_deg)
        self.assertLessEqual(abs(heave), first.wave_heave_m)

    def test_training_and_held_out_motion_distributions_are_disjoint(self) -> None:
        training = random_training_motion_profile(101)
        medium = random_evaluation_motion_profile(101, "medium")
        hard = random_evaluation_motion_profile(101, "hard")
        training_speeds = [float(np.hypot(*training.velocity_at(t))) for t in np.linspace(0.0, 50.0, 251)]
        medium_speeds = [float(np.hypot(*medium.velocity_at(t))) for t in np.linspace(0.0, 50.0, 251)]
        hard_speeds = [float(np.hypot(*hard.velocity_at(t))) for t in np.linspace(0.0, 50.0, 251)]
        self.assertLess(max(training_speeds), min(medium_speeds))
        self.assertLess(max(medium_speeds), min(hard_speeds))
        train_distance = float(np.hypot(*training.position_at(50.0)))
        hard_distance = float(np.hypot(*hard.position_at(50.0)))
        self.assertGreater(hard_distance, train_distance * 4.0)

    def test_moving_qr_world_enables_seeded_curved_path_and_deck_wave(self) -> None:
        model = (
            PROJECT_ROOT / "PX4-Autopilot/Tools/simulation/gz/models/moving_qr_landing_pad/model.sdf"
        ).read_text(encoding="utf-8")
        world = (
            PROJECT_ROOT / "PX4-Autopilot/Tools/simulation/gz/worlds/aruco_moving_qr.sdf"
        ).read_text(encoding="utf-8")
        self.assertIn("<static>false</static>", model)
        self.assertIn("libMovingPlatformController.so", model)
        self.assertIn("<trajectory_enabled>true</trajectory_enabled>", model)
        self.assertIn("<wave_enabled>true</wave_enabled>", model)
        self.assertIn("<noise_enabled>false</noise_enabled>", model)
        self.assertIn("<motion_start_delay_s>10.0</motion_start_delay_s>", model)
        self.assertIn("model://moving_qr_landing_pad", world)

    def test_zero_residual_tracks_the_moving_pad_to_a_guarded_landing(self) -> None:
        environment = QrPrecisionLandingEnv(seed=39)
        observation, _ = environment.reset(seed=39)
        done = False
        info: dict[str, float] = {}
        while not done:
            observation, _, terminated, truncated, info = environment.step(np.zeros(2, dtype=np.float32))
            done = terminated or truncated
        self.assertGreater(info["target_speed_mps"], 0.0)
        self.assertIn("deck_heave_m", info)
        self.assertEqual(info["success"], 1.0)


if __name__ == "__main__":
    unittest.main()
