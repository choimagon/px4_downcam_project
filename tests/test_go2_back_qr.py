import unittest
import ast
import inspect
import textwrap
from pathlib import Path

import mujoco
import numpy as np

from landing_rl.go2_qr_environment import (
    CAMERA_FRAME_PERIOD_S,
    CAMERA_HFOV_RAD,
    CAMERA_NEAR_M,
    CAMERA_PNP_ROTATION_NOISE_BASE_DEG,
    CAMERA_PNP_ROTATION_NOISE_PER_M_DEG,
    CAMERA_VFOV_RAD,
    DRONE_OBSERVATION_NAMES,
    ESTIMATOR_PERIOD_S,
    FINAL_BLIND_DESCENT_SPEED_MPS,
    FINAL_APPROACH_MEMORY_S,
    FINAL_RETRY_BLIND_DESCENT_SPEED_MPS,
    IMU_IMPACT_DELTA_MPS2,
    IMU_IMPACT_MAX_VISUAL_HEIGHT_M,
    IMU_SETTLE_TIME_S,
    IMU_SETTLE_THRUST_FRACTION,
    LANDING_POLICY_RESIDUAL_SPEED_MPS,
    LANDING_POLICY_TRAINING_RESIDUAL_SPEED_MPS,
    GO2_PROFILES,
    QR_INK_RENDER_CLEARANCE_M,
    QR_LANDING_SURFACE_TOP_M,
    QR_MIN_DETECT_PX,
    QR_CENTER_SITE_Z_M,
    QR_PRINT_TOP_M,
    SEARCH_ALTITUDE_WORLD_M,
    X500_NOMINAL_TOUCHDOWN_RELATIVE_HEIGHT_M,
    X500_SKID_CENTER_BODY_Z_M,
    X500_SKID_HALF_SIZE_M,
    X500_SKID_LATERAL_OFFSET_M,
    X500_VISUAL_SKID_BOTTOM_BODY_Z_M,
    Go2BackQrLandingEnv,
)


class Go2BackQrEnvironmentTest(unittest.TestCase):
    def test_drone_policy_contract_has_only_real_onboard_state(self) -> None:
        self.assertEqual(
            DRONE_OBSERVATION_NAMES,
            (
                "qr_center_u", "qr_center_v", "qr_pnp_depth", "qr_detected",
                "qr_center_rate_u", "qr_center_rate_v", "drone_vertical_velocity",
            ),
        )
        forbidden_words = ("go2", "pad", "base", "mount")
        self.assertFalse(any(word in name for name in DRONE_OBSERVATION_NAMES for word in forbidden_words))
        self.assertFalse(any("contact" in name or "gear" in name for name in DRONE_OBSERVATION_NAMES))

        # Also guard the controller's executable attribute accesses, so a
        # future refactor cannot silently reintroduce privileged target state.
        source = "\n".join(
            textwrap.dedent(inspect.getsource(method))
            for method in (Go2BackQrLandingEnv._observation, Go2BackQrLandingEnv._drone_control)
        )
        tree = ast.parse(source)
        attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        self.assertTrue(
            attributes.isdisjoint(
                {
                    "_horizontal_error", "_relative_altitude", "_pad_velocity",
                    "_pad_vertical_velocity", "pad_position", "base_position",
                    "mount_id", "base_id", "_onboard_gear_contact_count",
                    "_offline_sim_landing_skid_contact_count",
                }
            )
        )

    def test_observation_is_invariant_to_privileged_go2_pad_telemetry(self) -> None:
        env = Go2BackQrLandingEnv(seed=143, difficulty="easy")
        for sensor_name in (
            "drone_gps_position", "drone_gps_velocity", "drone_attitude", "drone_gyro",
            "drone_accelerometer",
        ):
            self.assertGreaterEqual(env.model.sensor(sensor_name).id, 0)
        with self.assertRaises(KeyError):
            env.model.sensor("gear_contact_fl")
        env._qr_detected = True
        env._qr_center_norm[:] = (0.21, -0.17)
        env._qr_depth = 2.4
        env._qr_center_rate[:] = (0.33, -0.22)
        before = env._observation().copy()
        env._pad_velocity[:] = (99.0, -99.0)
        env._pad_vertical_velocity = 77.0
        env.data.qpos[env.go2_qposadr] += np.linspace(-0.4, 0.4, 12)
        after = env._observation().copy()
        self.assertTrue(np.array_equal(before, after))
        env.close()

    @staticmethod
    def _place_drone_at_camera_depth(
        env: Go2BackQrLandingEnv, depth_m: float, *, offset_x_m: float = 0.0, offset_y_m: float = 0.0
    ) -> None:
        marker = env.pad_position.copy()
        address = env.drone_qposadr
        env.data.qpos[address:address + 7] = (
            marker[0] + offset_x_m,
            marker[1] + offset_y_m,
            marker[2] + depth_m + 0.065,
            1.0, 0.0, 0.0, 0.0,
        )
        env.data.qvel[env.drone_dofadr:env.drone_dofadr + 6] = 0.0
        mujoco.mj_forward(env.model, env.data)
        env._update_onboard_estimator(force=True)

    @staticmethod
    def _prime_control_cache(env: Go2BackQrLandingEnv, mode: str) -> None:
        """Install identical onboard/cache state without reading the Go2."""
        env.data.time = 6.0
        env._estimated_position[:] = (1.25, -0.75, SEARCH_ALTITUDE_WORLD_M)
        env._estimated_velocity[:] = (0.08, -0.03, 0.02)
        env._estimated_rotation[:] = np.eye(3)
        env._estimated_angular_velocity[:] = (0.01, -0.02, 0.03)
        env._aligned_streak = 0
        env._approach_recovery = False
        env._qr_center_norm[:] = (0.10, -0.20)
        env._qr_center_rate[:] = (0.05, -0.04)
        env._qr_translation_body[:] = (0.35, -0.18, -0.90)
        env._qr_relative_velocity_world[:] = (0.12, -0.08, 0.0)
        env._qr_target_velocity_world[:] = (0.72, 0.11, -0.02)
        env._qr_target_position_world[:] = env._estimated_position + np.array((0.04, -0.03, -0.18))
        env._qr_target_rotation_world[:] = np.eye(3)
        env._search_started = 0.0
        env._search_altitude = SEARCH_ALTITUDE_WORLD_M
        if mode == "search":
            env._qr_detected = False
            env._last_qr_seen_time = float("-inf")
            env._final_approach = False
            env._landing_committed = False
        elif mode == "tracking":
            env._qr_detected = True
            env._last_qr_seen_time = float(env.data.time)
            env._final_approach = False
            env._landing_committed = False
        elif mode == "final":
            env._qr_detected = False
            env._last_qr_seen_time = float(env.data.time) - 0.05
            env._final_approach = True
            env._landing_committed = True
        else:
            raise ValueError(f"unknown control mode: {mode}")
        env.data.xfrc_applied[env.drone_id] = 0.0

    @staticmethod
    def _mutate_hidden_go2_state(env: Go2BackQrLandingEnv) -> None:
        """Change every hidden Go2 state family while preserving sensor caches."""
        env.data.qpos[:7] = (8.0, -5.0, 0.55, 0.9238795, 0.0, 0.0, 0.3826834)
        env.data.qvel[:6] = (2.0, -1.0, 0.4, 0.3, -0.2, 0.5)
        env.data.qpos[env.go2_qposadr] += np.linspace(-0.7, 0.7, 12)
        env.data.qvel[env.go2_dofadr] = np.linspace(4.0, -4.0, 12)
        mujoco.mj_forward(env.model, env.data)

    def test_no_landing_gear_sensor_exists(self) -> None:
        env = Go2BackQrLandingEnv(seed=19, difficulty="easy")
        sensor_names = {
            env.model.sensor(sensor_id).name.lower()
            for sensor_id in range(env.model.nsensor)
        }
        self.assertFalse(any(token in name for name in sensor_names for token in ("gear", "touch", "load", "contact")))
        env.close()

    def test_missed_qr_masks_every_camera_feature(self) -> None:
        env = Go2BackQrLandingEnv(seed=20, difficulty="easy")
        env._qr_detected = False
        env._qr_center_norm[:] = (0.6, -0.4)
        env._qr_depth = 2.7
        env._qr_center_rate[:] = (1.4, -1.2)
        self.assertTrue(np.array_equal(env._observation()[:6], np.zeros(6, dtype=np.float32)))
        env.close()

    def test_pnp_uses_camera_optical_origin_and_stock_near_clip(self) -> None:
        env = Go2BackQrLandingEnv(seed=21, difficulty="easy")
        env._dropout = False
        self._place_drone_at_camera_depth(env, 0.11)
        env._update_qr_camera_measurement(force=True)
        actual_depth = float(env.data.cam_xpos[env.down_camera_id, 2] - env.pad_position[2])
        self.assertTrue(env._qr_detected)
        self.assertAlmostEqual(actual_depth, 0.11, delta=0.002)
        self.assertAlmostEqual(env._qr_depth, actual_depth, delta=0.01)
        self._place_drone_at_camera_depth(env, 0.09)
        env._update_qr_camera_measurement(force=True)
        self.assertFalse(env._qr_detected)
        self.assertEqual(env._qr_depth, 0.0)
        self.assertEqual(CAMERA_NEAR_M, 0.10)
        env.close()

    def test_pnp_rotation_is_noisy_camera_measurement_and_sample_hold(self) -> None:
        env = Go2BackQrLandingEnv(seed=211, difficulty="easy")
        try:
            env._dropout = False
            self._place_drone_at_camera_depth(env, 0.50)
            env._update_qr_camera_measurement(force=True)
            self.assertTrue(env._qr_detected)
            exact_body_rotation = (
                env._onboard_rotation().T
                @ env.data.xmat[env.mount_id].reshape(3, 3)
            )
            rotation_delta = exact_body_rotation.T @ env._qr_rotation_body
            angle_deg = np.degrees(
                np.arccos(np.clip((np.trace(rotation_delta) - 1.0) / 2.0, -1.0, 1.0))
            )
            self.assertGreater(angle_deg, 1.0e-6)
            max_sigma_deg = (
                CAMERA_PNP_ROTATION_NOISE_BASE_DEG
                + CAMERA_PNP_ROTATION_NOISE_PER_M_DEG * 0.50
            )
            self.assertLessEqual(angle_deg, np.sqrt(3.0) * 3.0 * max_sigma_deg + 1.0e-6)
            np.testing.assert_allclose(
                env._qr_target_rotation_world,
                env._onboard_rotation() @ env._qr_rotation_body,
                atol=1.0e-12,
            )
            cache_before = env._qr_target_rotation_world.copy()
            env.data.qpos[3:7] = (
                np.cos(np.radians(8.0) / 2.0), 0.0, 0.0,
                np.sin(np.radians(8.0) / 2.0),
            )
            mujoco.mj_forward(env.model, env.data)
            env._update_qr_camera_measurement()
            np.testing.assert_array_equal(env._qr_target_rotation_world, cache_before)
            source = inspect.getsource(Go2BackQrLandingEnv._update_qr_camera_measurement)
            self.assertNotIn("_qr_target_rotation_world[:] = marker_rotation", source)
        finally:
            env.close()

    def test_stock_camera_aspect_fov_and_marker_pixel_threshold(self) -> None:
        env = Go2BackQrLandingEnv(seed=22, difficulty="easy")
        env._dropout = False
        self.assertAlmostEqual(float(env.model.cam_fovy[env.down_camera_id]), np.degrees(CAMERA_VFOV_RAD), delta=0.02)
        self.assertGreater(CAMERA_HFOV_RAD, CAMERA_VFOV_RAD)
        self.assertEqual(QR_MIN_DETECT_PX, 20.0)
        self._place_drone_at_camera_depth(env, 1.50, offset_x_m=1.50)
        env._update_qr_camera_measurement(force=True)
        self.assertTrue(env._qr_detected)  # inside horizontal FOV
        self._place_drone_at_camera_depth(env, 1.50, offset_y_m=1.50)
        env._update_qr_camera_measurement(force=True)
        self.assertFalse(env._qr_detected)  # outside narrower vertical FOV
        self._place_drone_at_camera_depth(env, 7.0)
        env._update_qr_camera_measurement(force=True)
        self.assertFalse(env._qr_detected)  # 23 cm marker is below 20 px
        env.close()

    def test_camera_and_estimator_are_sample_and_hold(self) -> None:
        env = Go2BackQrLandingEnv(seed=23, difficulty="easy")
        env._dropout = False
        self._place_drone_at_camera_depth(env, 1.0, offset_x_m=0.20)
        env.data.time = 0.0
        env._next_estimator_time = 0.0
        env._next_camera_time = 0.0
        env._update_onboard_estimator(force=True)
        env._update_qr_camera_measurement(force=True)
        first_center = env._qr_center_norm.copy()
        first_estimate = env._onboard_position()

        address = env.drone_qposadr
        env.data.qpos[address] -= 0.40
        mujoco.mj_forward(env.model, env.data)
        env.data.time = 0.005
        env._update_onboard_estimator()
        env._update_qr_camera_measurement()
        self.assertTrue(np.array_equal(first_center, env._qr_center_norm))
        self.assertTrue(np.array_equal(first_estimate, env._onboard_position()))

        env.data.time = max(CAMERA_FRAME_PERIOD_S, ESTIMATOR_PERIOD_S) + 0.001
        env._update_onboard_estimator()
        env._update_qr_camera_measurement()
        self.assertFalse(np.array_equal(first_center, env._qr_center_norm))
        self.assertGreater(abs(float(first_estimate[0] - env._onboard_position()[0])), 0.25)
        env.close()

    def test_final_approach_reacquisition_preserves_camera_velocity_memory(self) -> None:
        env = Go2BackQrLandingEnv(seed=231, difficulty="hard")
        try:
            env._dropout = False
            self._place_drone_at_camera_depth(env, 0.11)
            env.data.time = 3.0
            env._next_camera_time = 0.0
            env._final_approach = True
            env._last_qr_seen_time = env.data.time - 0.05
            env._previous_qr_valid = False
            remembered = np.array((0.83, -0.21, 0.04), dtype=np.float64)
            env._qr_target_velocity_world[:] = remembered
            env._update_qr_camera_measurement(force=True)
            self.assertTrue(env._qr_detected)
            np.testing.assert_array_equal(env._qr_target_velocity_world, remembered)
            self.assertLess(env.data.time - env._last_qr_seen_time, FINAL_APPROACH_MEMORY_S)
            self.assertEqual(LANDING_POLICY_RESIDUAL_SPEED_MPS, 0.001)
            self.assertEqual(LANDING_POLICY_TRAINING_RESIDUAL_SPEED_MPS, 0.002)
            self.assertEqual(CAMERA_PNP_ROTATION_NOISE_BASE_DEG, 0.15)
            self.assertEqual(CAMERA_PNP_ROTATION_NOISE_PER_M_DEG, 0.03)
            self.assertEqual(IMU_SETTLE_TIME_S, 0.35)
            self.assertEqual(FINAL_BLIND_DESCENT_SPEED_MPS, 0.16)
            self.assertEqual(FINAL_RETRY_BLIND_DESCENT_SPEED_MPS, 0.14)
        finally:
            env.close()

    def test_training_exploration_and_deployment_safety_envelopes_are_explicit(self) -> None:
        deployed = Go2BackQrLandingEnv(seed=233, difficulty="easy")
        training = Go2BackQrLandingEnv(
            seed=233,
            difficulty="easy",
            policy_residual_speed_mps=LANDING_POLICY_TRAINING_RESIDUAL_SPEED_MPS,
        )
        try:
            self.assertEqual(deployed.policy_residual_speed_mps, 0.001)
            self.assertEqual(training.policy_residual_speed_mps, 0.002)
            with self.assertRaises(ValueError):
                Go2BackQrLandingEnv(policy_residual_speed_mps=0.011)
        finally:
            deployed.close()
            training.close()

    def test_imu_landing_state_uses_no_contact_sensor(self) -> None:
        env = Go2BackQrLandingEnv(seed=232, difficulty="hard")
        try:
            env._final_approach = True
            env._last_qr_seen_time = float(env.data.time)
            env._estimated_velocity[:] = 0.0
            env._estimated_rotation[:] = np.eye(3, dtype=np.float64)
            env._qr_detected = True
            env._qr_translation_body[:] = (
                0.0, 0.0, -X500_NOMINAL_TOUCHDOWN_RELATIVE_HEIGHT_M
            )
            original_sensor = env._sensor
            env._commanded_specific_force_body_z = 18.0
            env._sensor = lambda name: (
                np.array((0.0, 0.0, 18.0))
                if name == "drone_accelerometer"
                else original_sensor(name)
            )
            env._update_imu_landing_state()
            self.assertFalse(env._imu_impact_latched)
            env._sensor = lambda name: (
                np.array((0.0, 0.0, 18.0 + IMU_IMPACT_DELTA_MPS2 + 0.5))
                if name == "drone_accelerometer"
                else original_sensor(name)
            )
            env._update_imu_landing_state()
            self.assertTrue(env._imu_impact_latched)
            self.assertLessEqual(
                X500_NOMINAL_TOUCHDOWN_RELATIVE_HEIGHT_M,
                IMU_IMPACT_MAX_VISUAL_HEIGHT_M,
            )
            source = textwrap.dedent(inspect.getsource(Go2BackQrLandingEnv._update_imu_landing_state))
            tree = ast.parse(source)
            attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
            self.assertTrue(
                attributes.isdisjoint(
                    {
                        "_contact_calibration",
                        "_offline_sim_landing_skid_contact_count",
                        "_offline_sim_landing_normal_force",
                        "_offline_sim_max_contact_penetration",
                        "contact",
                        "ncon",
                    }
                )
            )
        finally:
            env.close()

    def test_imu_settle_reduces_collective_instead_of_bouncing_skids(self) -> None:
        env = Go2BackQrLandingEnv(seed=235, difficulty="hard")
        try:
            self._prime_control_cache(env, "tracking")
            env._landing_committed = True
            env._final_approach = True
            env._imu_impact_latched = True
            env._qr_translation_body[:] = (
                0.01, -0.01, -X500_NOMINAL_TOUCHDOWN_RELATIVE_HEIGHT_M
            )
            env._drone_control(np.zeros(2), update_alignment=False)
            hover_force = env.drone_mass * abs(float(env.model.opt.gravity[2]))
            self.assertAlmostEqual(IMU_SETTLE_THRUST_FRACTION, 0.88)
            self.assertAlmostEqual(
                float(env.data.xfrc_applied[env.drone_id, 2]),
                IMU_SETTLE_THRUST_FRACTION * hover_force,
                places=10,
            )
        finally:
            env.close()

    def test_search_and_final_approach_use_no_contact_sensor(self) -> None:
        env = Go2BackQrLandingEnv(seed=24, difficulty="easy")
        env._estimated_position[:] = (4.0, -2.0, SEARCH_ALTITUDE_WORLD_M)
        env._estimated_velocity[:] = 0.0
        env._estimated_rotation[:] = np.eye(3)
        env._estimated_angular_velocity[:] = 0.0
        env._qr_detected = False
        env._final_approach = False
        env._drone_control(np.zeros(2), update_alignment=True)
        self.assertGreater(np.linalg.norm(env.data.xfrc_applied[env.drone_id, :2]), 0.1)

        env._landing_committed = True
        env._final_approach = True
        env._last_qr_seen_time = float(env.data.time)
        env._qr_target_position_world[:] = env._estimated_position + np.array(
            [0.01, -0.01, -(X500_NOMINAL_TOUCHDOWN_RELATIVE_HEIGHT_M + 0.03)]
        )
        env._qr_target_velocity_world[:] = (0.7, 0.0, 0.0)
        env._drone_control(np.zeros(2), update_alignment=True)
        hover_force = env.drone_mass * abs(float(env.model.opt.gravity[2]))
        self.assertLess(float(env.data.xfrc_applied[env.drone_id, 2]), hover_force)
        self.assertTrue(env._final_approach)
        env.close()

    def test_search_tracking_and_final_commands_ignore_hidden_go2_state(self) -> None:
        action = np.array((0.31, -0.27), dtype=np.float32)
        for mode in ("search", "tracking", "final"):
            with self.subTest(mode=mode):
                reference = Go2BackQrLandingEnv(seed=91, difficulty="easy")
                hidden_changed = Go2BackQrLandingEnv(seed=92, difficulty="easy")
                try:
                    self._prime_control_cache(reference, mode)
                    self._prime_control_cache(hidden_changed, mode)
                    self._mutate_hidden_go2_state(hidden_changed)
                    reference._drone_control(action, update_alignment=True)
                    hidden_changed._drone_control(action, update_alignment=True)
                    np.testing.assert_array_equal(
                        reference.data.xfrc_applied[reference.drone_id],
                        hidden_changed.data.xfrc_applied[hidden_changed.drone_id],
                    )
                    self.assertEqual(
                        (
                            reference._aligned_streak,
                            reference._landing_committed,
                            reference._final_approach,
                            reference._approach_recovery,
                        ),
                        (
                            hidden_changed._aligned_streak,
                            hidden_changed._landing_committed,
                            hidden_changed._final_approach,
                            hidden_changed._approach_recovery,
                        ),
                    )
                finally:
                    reference.close()
                    hidden_changed.close()

    def test_control_wrench_is_deterministic_for_identical_sensor_cache_and_action(self) -> None:
        env = Go2BackQrLandingEnv(seed=93, difficulty="easy")
        action = np.array((-0.24, 0.37), dtype=np.float32)
        try:
            for mode in ("search", "tracking", "final"):
                with self.subTest(mode=mode):
                    self._prime_control_cache(env, mode)
                    env._drone_control(action, update_alignment=True)
                    first = env.data.xfrc_applied[env.drone_id].copy()
                    first_flags = (
                        env._aligned_streak,
                        env._landing_committed,
                        env._final_approach,
                        env._approach_recovery,
                    )

                    # RNG belongs to sensor noise and physical wind.  It must
                    # not alter a controller output when its inputs are fixed.
                    env.np_random.normal(size=128)
                    self._prime_control_cache(env, mode)
                    env._drone_control(action, update_alignment=True)
                    np.testing.assert_array_equal(first, env.data.xfrc_applied[env.drone_id])
                    self.assertEqual(
                        first_flags,
                        (
                            env._aligned_streak,
                            env._landing_committed,
                            env._final_approach,
                            env._approach_recovery,
                        ),
                    )
        finally:
            env.close()

    def test_rl_residual_cannot_push_away_or_act_inside_final_cutoff(self) -> None:
        zero = Go2BackQrLandingEnv(seed=233, difficulty="hard")
        guarded = Go2BackQrLandingEnv(seed=234, difficulty="hard")
        try:
            for mode, action in (
                ("tracking", np.array((-0.35, 0.18), dtype=np.float32)),
                ("final", np.array((1.0, -1.0), dtype=np.float32)),
            ):
                with self.subTest(mode=mode):
                    self._prime_control_cache(zero, mode)
                    self._prime_control_cache(guarded, mode)
                    zero._drone_control(np.zeros(2, dtype=np.float32), update_alignment=True)
                    guarded._drone_control(action, update_alignment=True)
                    np.testing.assert_allclose(
                        zero.data.xfrc_applied[zero.drone_id],
                        guarded.data.xfrc_applied[guarded.drone_id],
                        atol=1.0e-12,
                        rtol=0.0,
                    )
        finally:
            zero.close()
            guarded.close()

    def test_nonterminal_reward_does_not_read_landing_skid_contact_count(self) -> None:
        """Changing an offline contact report below touchdown must not shape reward."""
        no_contact = Go2BackQrLandingEnv(seed=94, difficulty="easy")
        two_contacts = Go2BackQrLandingEnv(seed=94, difficulty="easy")
        try:
            no_contact._contact_calibration = lambda: (0, 0.0, 0.0)
            two_contacts._contact_calibration = lambda: (2, 123.0, 0.02)
            action = np.array((0.13, -0.21), dtype=np.float32)
            observation_a, reward_a, terminated_a, truncated_a, info_a = no_contact.step(action)
            observation_b, reward_b, terminated_b, truncated_b, info_b = two_contacts.step(action)
            np.testing.assert_array_equal(observation_a, observation_b)
            self.assertEqual(reward_a, reward_b)
            self.assertEqual((terminated_a, truncated_a), (terminated_b, truncated_b))
            self.assertEqual(info_a["success"], 0.0)
            self.assertEqual(info_b["success"], 0.0)
            self.assertEqual(info_a["offline_sim_landing_skid_contacts"], 0.0)
            self.assertEqual(info_b["offline_sim_landing_skid_contacts"], 2.0)
        finally:
            no_contact.close()
            two_contacts.close()

    def test_qr_simulator_truth_access_is_explicitly_allowlisted(self) -> None:
        tree = ast.parse(textwrap.dedent(inspect.getsource(Go2BackQrLandingEnv)))
        class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        methods = {
            node.name: node
            for node in class_node.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        allowlist = {
            # ID lookup is not a measurement.  Raw QR pose is measured only
            # by the camera boundary; pad_position is a simulator accessor.
            "qr_site_id": {"__init__", "pad_position", "_update_qr_camera_measurement"},
            "mount_id": {"__init__", "_update_qr_camera_measurement"},
            # The accessor and derived truth metrics are restricted to reset
            # randomization and simulator-only reward/scoring in step().
            "pad_position": {"_relative_altitude", "_horizontal_error", "reset", "step"},
            "_relative_altitude": {"reset", "step"},
            "_horizontal_error": {"step"},
            "_pad_velocity": {"__init__", "reset", "step"},
            "_pad_vertical_velocity": {"__init__", "reset", "step"},
        }
        for attribute, allowed_methods in allowlist.items():
            actual_methods = {
                name
                for name, method in methods.items()
                if any(
                    isinstance(node, ast.Attribute) and node.attr == attribute
                    for node in ast.walk(method)
                )
            }
            self.assertEqual(actual_methods, allowed_methods, msg=f"unexpected use of self.{attribute}")

    def test_official_go2_qr_mount_is_rigid_and_clears_contact_landing(self) -> None:
        locomotion_model = Path(__file__).resolve().parents[1] / "models" / "go2_legged_loco_ppo.zip"
        self.assertTrue(locomotion_model.exists())
        env = Go2BackQrLandingEnv(seed=100, difficulty="easy", locomotion_model=locomotion_model)
        mount_id = env.model.body("qr_mount").id
        self.assertEqual(env.model.body_parentid[mount_id], env.base_id)
        self.assertEqual(env.model.body_jntnum[mount_id], 0)  # fixed child: no plate wobble/freejoint
        self.assertEqual(env.model.nu, 12)  # official Go2 has 12 torque-controlled joints
        self.assertTrue(np.allclose(env.model.geom_friction[env.landing_surface_id], (0.95, 0.015, 0.001)))
        # The 23 cm QR is printed on a reduced 36 cm *physical* plate.  The
        # ink receives only a three-micrometre render clearance over the
        # colliding top so the down-camera cannot suffer coplanar z-fighting.
        qr = env.model.geom("qr_black_nw")
        plate = env.model.geom("landing_surface")
        self.assertAlmostEqual(float(qr.size[0]), 0.0225)
        self.assertAlmostEqual(float(qr.size[1]), 0.0225)
        self.assertTrue(np.allclose(plate.size[:2], (0.18, 0.18)))
        # qr_deck is the visible twin of the transparent collision board;
        # their top planes are equal so the visible QR deck is the physical
        # landing surface without rendering it twice.
        visible_deck = env.model.geom("qr_deck")
        self.assertEqual(float(plate.rgba[3]), 0.0)
        self.assertEqual(float(visible_deck.rgba[3]), 1.0)
        self.assertEqual(int(env.model.geom_condim[plate.id]), 3)
        self.assertAlmostEqual(
            float(qr.pos[2] + qr.size[2] - plate.pos[2] - plate.size[2]),
            QR_INK_RENDER_CLEARANCE_M,
            delta=1.0e-9,
        )
        self.assertAlmostEqual(float(plate.pos[2] + plate.size[2]), QR_LANDING_SURFACE_TOP_M, delta=1.0e-9)
        self.assertAlmostEqual(
            float(visible_deck.pos[2] + visible_deck.size[2]),
            float(plate.pos[2] + plate.size[2]),
            delta=1.0e-9,
        )
        skid = env.model.geom("drone_skid_left")
        self.assertTrue(np.allclose(skid.pos[:2], (0.0, 0.132)))

        _, _ = env.reset(seed=100)
        final_info = {}
        for _ in range(env.max_steps):
            _, _, terminated, truncated, final_info = env.step(np.zeros(2, dtype=np.float32))
            if terminated or truncated:
                break
        self.assertEqual(final_info["success"], 1.0)
        self.assertEqual(final_info["go2_fall"], 0.0)
        self.assertEqual(final_info["offline_sim_landing_skid_contacts"], 2.0)
        self.assertLess(final_info["offline_sim_max_contact_penetration_m"], 0.001)
        self.assertGreater(final_info["go2_path_distance_m"], 0.5)
        self.assertEqual(final_info["locomotion_backend"], "legged-loco-mujoco-ppo")

    def test_x500_visual_skids_are_visible_contact_objects_on_the_qr_plane(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        frame_mesh = project_root / "assets" / "mujoco_x500" / "x500_frame.obj"
        raw_vertex_z = []
        with frame_mesh.open(encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("v "):
                    raw_vertex_z.append(float(line.split()[3]))
        self.assertTrue(raw_vertex_z)

        # x500_frame_visual is translated +25 mm in the MJCF.  Its rendered
        # skid sole and both physical stock skid rails must share one body-Z plane.
        visual_skid_sole_z = min(raw_vertex_z) + 0.025
        self.assertAlmostEqual(
            visual_skid_sole_z,
            X500_VISUAL_SKID_BOTTOM_BODY_Z_M,
            delta=1.0e-9,
        )

        env = Go2BackQrLandingEnv(seed=101, difficulty="easy")
        try:
            expected_xy = {
                "drone_skid_left": (0.0, X500_SKID_LATERAL_OFFSET_M),
                "drone_skid_right": (0.0, -X500_SKID_LATERAL_OFFSET_M),
            }
            for name, xy in expected_xy.items():
                skid = env.model.geom(name)
                self.assertTrue(np.allclose(skid.pos[:2], xy))
                self.assertEqual(
                    int(env.model.geom_type[skid.id]),
                    int(mujoco.mjtGeom.mjGEOM_BOX),
                )
                self.assertTrue(np.allclose(skid.size, X500_SKID_HALF_SIZE_M))
                # MuJoCo stores box half-extents; these are the exact full
                # dimensions of each official PX4 SDF skid collision rail.
                self.assertTrue(np.allclose(2.0 * skid.size, (0.25, 0.015, 0.015)))
                self.assertAlmostEqual(
                    float(skid.pos[2]),
                    X500_SKID_CENTER_BODY_Z_M,
                )
                contact_sole_z = float(skid.pos[2] - skid.size[2])
                self.assertLessEqual(abs(contact_sole_z - visual_skid_sole_z), 0.001)
                # The visual landing sole is the contact object itself, not
                # an invisible proxy under a different rendered model.
                self.assertEqual(float(skid.rgba[3]), 1.0)
                self.assertEqual(int(env.model.geom_condim[skid.id]), 3)

            for name in ("mono_cam_housing", "mono_cam_lens"):
                self.assertEqual(float(env.model.geom(name).rgba[3]), 0.0)
            self.assertEqual(float(env.model.site("qr_center").rgba[3]), 0.0)

            # qr_center is 1.6 mm above the physical QR board top.  The ink
            # is a 3 μm render layer only, so touchdown uses the board plane.
            self.assertAlmostEqual(
                X500_NOMINAL_TOUCHDOWN_RELATIVE_HEIGHT_M,
                QR_LANDING_SURFACE_TOP_M - QR_CENTER_SITE_Z_M - visual_skid_sole_z,
                delta=1.0e-9,
            )

            # The imported STL has an offset mesh origin.  The four propeller
            # geoms must be translated by the PX4 SDF local visual pose so
            # their rendered centres sit over their motor/rotor axes.
            rotor_pairs = (
                ("propeller_front_right", "rotor_axis_front_right"),
                ("propeller_rear_left", "rotor_axis_rear_left"),
                ("propeller_front_left", "rotor_axis_front_left"),
                ("propeller_rear_right", "rotor_axis_rear_right"),
            )
            mujoco.mj_forward(env.model, env.data)
            for propeller_name, axis_name in rotor_pairs:
                propeller = env.model.geom(propeller_name)
                rotor_site = env.model.site(axis_name)
                self.assertLessEqual(
                    float(np.linalg.norm(env.data.geom_xpos[propeller.id, :2] - env.data.site_xpos[rotor_site.id, :2])),
                    0.0001,
                    msg=f"{propeller_name} must be centred over {axis_name}",
                )
        finally:
            env.close()

    def test_difficulty_is_defined_by_speed_and_route_turn_complexity(self) -> None:
        easy, medium, hard = (GO2_PROFILES[name] for name in ("easy", "medium", "hard"))
        self.assertLess(float(easy["path_speed"]), float(medium["path_speed"]))
        self.assertLess(float(medium["path_speed"]), float(hard["path_speed"]))
        self.assertLess(float(easy["turn_angle_rad"]), float(medium["turn_angle_rad"]))
        self.assertLess(float(medium["turn_angle_rad"]), float(hard["turn_angle_rad"]))
        self.assertLess(float(easy["turn_frequency_hz"]), float(medium["turn_frequency_hz"]))
        self.assertLess(float(medium["turn_frequency_hz"]), float(hard["turn_frequency_hz"]))
        self.assertEqual(easy["radius"], medium["radius"])
        self.assertEqual(medium["radius"], hard["radius"])


if __name__ == "__main__":
    unittest.main()
