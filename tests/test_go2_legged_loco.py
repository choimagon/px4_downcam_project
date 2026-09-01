import unittest
from unittest import mock

import numpy as np

import landing_rl.go2_legged_loco_environment as loco_environment
from landing_rl.go2_legged_loco_environment import (
    DEPLOYMENT_POLICY_ACTION_GAIN,
    JOINT_RESIDUAL_SCALE_RAD,
    STAND_POSE,
    Go2LeggedLocoEnv,
    _rpy,
)


class Go2LeggedLocoTest(unittest.TestCase):
    def test_legged_loco_state_contract_and_safe_zero_residual(self) -> None:
        env = Go2LeggedLocoEnv(
            seed=44,
            history_length=9,
            domain_randomization=False,
            sensor_noise=False,
        )
        observation, info = env.reset(seed=44)
        self.assertEqual(observation.shape, (450,))
        self.assertEqual(env.action_space.shape, (12,))
        self.assertEqual(env.model.nu, 12)
        self.assertEqual(info["payload_kg"], 0.22)
        self.assertGreaterEqual(float(env._command[0]), 0.35)
        self.assertLessEqual(float(env._command[0]), 1.20)
        final = {}
        heights = []
        slips = []
        assists = []
        matches = []
        for _ in range(300):
            _, _, terminated, truncated, final = env.step(np.zeros(12, dtype=np.float32))
            heights.append(final["base_height_m"])
            slips.append(final["stance_foot_slip_mps"])
            assists.append(final["assist_force_n"])
            matches.append(final["gait_contact_match"])
            if terminated or truncated:
                break
        self.assertEqual(final["fall"], 0.0)
        self.assertGreater(final["path_distance_m"], 2.0)
        self.assertGreater(final["base_up"], 0.98)
        self.assertGreater(float(np.mean(heights)), 0.29)
        self.assertLess(float(np.mean(heights)), 0.34)
        self.assertLess(float(np.mean(slips)), 0.25)
        self.assertEqual(float(np.max(assists)), 0.0)
        self.assertGreater(float(np.mean(matches)), 0.65)
        env.close()

    def test_root_wrench_is_zero_at_every_physics_step_and_torque_metric_matches(self) -> None:
        env = Go2LeggedLocoEnv(
            seed=52,
            history_length=0,
            max_steps=24,
            domain_randomization=False,
            sensor_noise=False,
        )
        env.reset(seed=52)
        original_mj_step = loco_environment.mujoco.mj_step
        observed_wrenches: list[np.ndarray] = []
        physics_ctrl: list[np.ndarray] = []

        def checked_mj_step(model: object, data: object) -> None:
            observed_wrenches.append(env.data.xfrc_applied[env.base_id, :6].copy())
            physics_ctrl.append(env.data.ctrl[env.actuator_ids].copy())
            original_mj_step(model, data)

        saw_saturation = False
        with mock.patch.object(loco_environment.mujoco, "mj_step", side_effect=checked_mj_step):
            for step_index in range(12):
                # A non-zero sentinel proves the controller clears all six root
                # wrench components before MuJoCo advances each control step.
                env.data.xfrc_applied[env.base_id, :6] = np.arange(1.0, 7.0)
                if step_index == 11:
                    # Force one deliberately saturated control sample so this
                    # test checks the positive path, not only the valid zero.
                    env.data.qpos[env.qposadr] = env._gait_target(float(env.data.time)) + 1.0
                    env.data.qvel[env.dofadr] = 0.0
                observed_wrenches.clear()
                physics_ctrl.clear()
                action = np.ones(12, dtype=np.float32) if step_index >= 5 else np.zeros(12, dtype=np.float32)
                _, _, terminated, truncated, info = env.step(action)

                self.assertEqual(len(observed_wrenches), env.physics_steps)
                self.assertTrue(all(np.array_equal(wrench, np.zeros(6)) for wrench in observed_wrenches))
                self.assertEqual(info["root_wrench_max_abs"], 0.0)

                limits = env.model.actuator_ctrlrange[env.actuator_ids]
                expected_saturation = max(
                    float(np.mean((ctrl <= limits[:, 0]) | (ctrl >= limits[:, 1])))
                    for ctrl in physics_ctrl
                )
                self.assertTrue(np.isfinite(info["torque_saturation_fraction"]))
                self.assertGreaterEqual(info["torque_saturation_fraction"], 0.0)
                self.assertLessEqual(info["torque_saturation_fraction"], 1.0)
                self.assertAlmostEqual(info["torque_saturation_fraction"], expected_saturation)
                saw_saturation = saw_saturation or expected_saturation > 0.0
                if terminated or truncated:
                    break

        self.assertTrue(saw_saturation, "the saturation metric was never exercised")
        env.close()

    def test_normalized_action_maps_once_to_point_18_rad_residual(self) -> None:
        env = Go2LeggedLocoEnv(
            seed=61,
            history_length=0,
            domain_randomization=False,
            sensor_noise=False,
        )
        env.reset(seed=61)
        self.assertAlmostEqual(JOINT_RESIDUAL_SCALE_RAD, 0.18)
        self.assertAlmostEqual(DEPLOYMENT_POLICY_ACTION_GAIN, 0.50)
        self.assertTrue(np.array_equal(env.action_space.low, -np.ones(12, dtype=np.float32)))
        self.assertTrue(np.array_equal(env.action_space.high, np.ones(12, dtype=np.float32)))

        time_s = float(env.data.time)
        reference = env._gait_target(time_s)
        next_reference = env._gait_target(time_s + float(env.model.opt.timestep))
        reference_velocity = (next_reference - reference) / float(env.model.opt.timestep)
        env.data.qpos[env.qposadr] = reference
        env.data.qvel[env.dofadr] = reference_velocity
        env._delay_steps = 0
        env._delayed_actions.clear()
        for _ in range(env._delayed_actions.maxlen or 7):
            env._delayed_actions.append(np.zeros(12, dtype=np.float64))

        action = np.linspace(-0.25, 0.25, 12, dtype=np.float64)
        env._apply_control(action, update_delay=True)
        expected_torque = 60.0 * JOINT_RESIDUAL_SCALE_RAD * action
        self.assertTrue(np.allclose(env.data.ctrl[env.actuator_ids], expected_torque, atol=1.0e-10))
        env.close()

    def test_single_observation_is_imu_joint_command_and_last_action_only(self) -> None:
        env = Go2LeggedLocoEnv(
            seed=73,
            history_length=0,
            domain_randomization=False,
            sensor_noise=False,
        )
        env.reset(seed=73)
        angular_velocity = np.array((0.31, -0.27, 0.19), dtype=np.float64)
        command = np.array((0.82, -0.14, 0.37), dtype=np.float64)
        joint_position = np.linspace(-0.18, 0.18, 12, dtype=np.float64)
        joint_velocity = np.linspace(-1.1, 1.1, 12, dtype=np.float64)
        last_action = np.linspace(0.45, -0.45, 12, dtype=np.float64)
        env.data.cvel[env.base_id, :3] = angular_velocity
        env._command[:] = command
        env.data.qpos[env.qposadr] = STAND_POSE + joint_position
        env.data.qvel[env.dofadr] = joint_velocity
        env._last_action[:] = last_action

        observation = env._single_observation()
        expected = np.concatenate(
            (
                angular_velocity,
                _rpy(env.data.xmat[env.base_id]),
                command,
                joint_position,
                joint_velocity,
                last_action,
            )
        )
        self.assertEqual(observation.shape, (45,))
        self.assertTrue(np.allclose(observation, expected))
        self.assertTrue(np.array_equal(observation[0:3], angular_velocity))
        self.assertTrue(np.array_equal(observation[6:9], command))
        self.assertTrue(np.allclose(observation[9:21], joint_position))
        self.assertTrue(np.allclose(observation[21:33], joint_velocity))
        self.assertTrue(np.array_equal(observation[33:45], last_action))
        env.close()


if __name__ == "__main__":
    unittest.main()
