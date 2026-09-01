"""MuJoCo port of the ``legged-loco`` Go2 low-level velocity task.

The referenced project is Isaac Lab-specific and does not release a Go2
checkpoint.  This environment keeps its public task contract in MuJoCo:

* 5 ms physics with a 4x decimation (20 ms low-level control),
* 45-dimensional proprioception in the documented term order,
* optional nine-frame history (450 state values, as the source wrapper emits),
* 12 normalized joint-position residual actions scaled once to 0.18 rad,
* 60/2 PD gains and a nominal four-control-step actuator-command delay.

The dorsal QR fixture is retained while training so the locomotion controller
experiences the payload it will carry during X500 landing.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .go2_qr_environment import GO2_STAND_POSE, build_go2_landing_xml
from .go2_terrain import (
    TERRAIN_SPEED_MULTIPLIER,
    TERRAIN_ROUTE_TARGET_X_M,
    configure_rough_terrain,
    terrain_geom_names,
    terrain_height_at,
    terrain_initial_pitch_rad,
    validate_rough_level,
    validate_terrain_task,
)


LEG_ORDER = ("FR", "FL", "RR", "RL")
JOINT_NAMES = tuple(f"{leg}_{joint}_joint" for leg in LEG_ORDER for joint in ("hip", "thigh", "calf"))
ACTUATOR_NAMES = tuple(f"{leg}_{joint}" for leg in LEG_ORDER for joint in ("hip", "thigh", "calf"))
STAND_POSE = GO2_STAND_POSE.copy()
# Map normalized PPO actions exactly once into joint-angle residuals.  The old
# double scaling saturated the policy while limiting it to only ±0.045 rad.
JOINT_RESIDUAL_SCALE_RAD = 0.18
# The retrained residual policy is deliberately deployed at half authority.
# A 30-episode randomized ablation showed this Pareto-dominates both the raw
# policy and zero residual in falls, velocity/yaw tracking, slip, gait timing,
# torque saturation and tilt.  The physical mapper itself remains a single
# 0.18 rad conversion; policy output is conditioned before entering it.
DEPLOYMENT_POLICY_ACTION_GAIN = 0.50
# Terrain actions remain a residual around the physical trot, but they must
# be active in both training and deployment.  The previous zero gate meant
# that a reported terrain policy never affected a single joint and left a
# phase-sensitive open-loop gait to handle the whole course.  Keep the
# authority large enough for terrain recovery (±54 mrad after the 0.18-rad
# mapper) and apply the same gain in the training environment and the 5-ms
# landing-scene bridge.  A 12% residual cannot counter a genuine foothold
# disturbance, so it was the direct cause of a policy that merely followed a
# failing open-loop trot on the old grade.
TERRAIN_RESIDUAL_POLICY_GAIN = 0.30
# This is an IMU-only course-heading reflex.  Values around 2.4 rad/s per rad
# over-corrected the 10% grade and eventually made the base spin; 1.2 keeps a
# contact-driven baseline upright long enough for the PPO residual to learn
# foothold recovery.
TERRAIN_YAW_FEEDBACK_GAIN = 1.20
# IMU yaw-rate damping prevents the position-only heading loop from turning a
# small foot-placement correction into a full-body spin on an inclined stance.
TERRAIN_YAW_RATE_DAMPING = 0.0
# High-level route tracking uses Go2's own odometry cross-track error only;
# it is not a terrain label and never enters the X500 policy.  It is kept
# outside the 450-D learned residual state as a deterministic safety prior.
TERRAIN_CROSS_TRACK_YAW_GAIN = 0.0
TERRAIN_YAW_HIP_GAIN = 0.0
TERRAIN_GAIT_RAMP_S = 0.80
TERRAIN_CADENCE_BASE_HZ = 2.75
TERRAIN_CADENCE_SPEED_GAIN = 0.68
TERRAIN_FORWARD_GAIN = 1.62
TERRAIN_FORWARD_LIMIT_MPS = 1.80
TERRAIN_STRIDE_LIMIT_M = 0.385
TERRAIN_STANCE_HEIGHT_M = 0.300
TERRAIN_SWING_LIFT_BASE_M = 0.118
TERRAIN_SWING_LIFT_SPEED_GAIN = 0.032
# Uphill stance needs the support polygon shifted 40 mm toward the grade.
# This was selected with a physical no-root-wrench sweep: it reaches the full
# 15 m course before the end of the now-extended collision surface.
TERRAIN_FOOT_CENTER_BIAS_M = -0.040
TERRAIN_PITCH_FOOT_CENTER_GAIN = -0.28
GAIT_DUTY_FACTOR = 0.58


def _terrain_gait_parameters(task: str) -> tuple[float, float, float, float, float, float]:
    """Return terrain-specific physical gait parameters.

    The continuous 80 mm heightfield needs a slower, more planted gait than
    the 10% inclined plane.  These are reference-foot trajectories only:
    base movement still comes exclusively from the four foot contacts and the
    12 joint torques.
    """
    if task == "rough":
        return (1.80, 0.68, 1.20, 1.80, 0.385, 0.5 * TERRAIN_FOOT_CENTER_BIAS_M)
    if task == "slope_up":
        return (2.75, 0.68, 1.62, 1.80, 0.385, TERRAIN_FOOT_CENTER_BIAS_M)
    return (
        TERRAIN_CADENCE_BASE_HZ,
        TERRAIN_CADENCE_SPEED_GAIN,
        TERRAIN_FORWARD_GAIN,
        TERRAIN_FORWARD_LIMIT_MPS,
        TERRAIN_STRIDE_LIMIT_M,
        0.0,
    )


def _rpy(rotation: np.ndarray) -> np.ndarray:
    """Return roll/pitch/yaw from MuJoCo's world-frame 3x3 orientation."""
    matrix = rotation.reshape(3, 3)
    pitch = math.asin(float(np.clip(-matrix[2, 0], -1.0, 1.0)))
    return np.array(
        [math.atan2(matrix[2, 1], matrix[2, 2]), pitch, math.atan2(matrix[1, 0], matrix[0, 0])],
        dtype=np.float64,
    )


def _sagittal_leg_ik(foot_x: float, foot_z: float) -> tuple[float, float]:
    """Return Go2 thigh/calf angles for a sagittal foot point.

    The official MuJoCo model has two 0.213 m links.  Solving the foot point
    directly keeps the stance foot on the floor; the old independent joint
    sinusoids shortened both links together and visibly made the robot skate.
    """
    link = 0.213
    radius_sq = float(foot_x * foot_x + foot_z * foot_z)
    cosine_knee = np.clip((radius_sq - 2.0 * link * link) / (2.0 * link * link), -0.999, 0.999)
    calf = -math.acos(float(cosine_knee))
    direction = math.atan2(-foot_x, -foot_z)
    thigh = direction - 0.5 * calf
    return thigh, calf


def _hermite_swing_x(u: float, rear: float, front: float, endpoint_slope: float) -> float:
    """Cubic swing return with stance-matched lift-off/touch-down velocity."""
    u2 = u * u
    u3 = u2 * u
    h00 = 2.0 * u3 - 3.0 * u2 + 1.0
    h10 = u3 - 2.0 * u2 + u
    h01 = -2.0 * u3 + 3.0 * u2
    h11 = u3 - u2
    return h00 * rear + h10 * endpoint_slope + h01 * front + h11 * endpoint_slope


def legged_loco_reference_target(
    time_s: float,
    command: np.ndarray,
    *,
    fast_terrain_gait: bool = False,
    terrain_task: str = "flat",
    body_rpy: np.ndarray | None = None,
    body_angular_velocity: np.ndarray | None = None,
    course_lateral_error_m: float = 0.0,
    desired_pitch_rad: float = 0.0,
) -> np.ndarray:
    """Raibert-style diagonal trot prior used for training and inference.

    Stance feet travel backward at the commanded body speed, making their
    world velocity approximately zero.  Diagonal pairs use a 58% duty factor
    for a short double-support interval, while swing feet follow a smooth
    rear-to-front arc.  Cadence and stride are coupled, so the visible step
    rate remains synchronized with 0.7--1.1 m/s route motion.
    """
    targets = STAND_POSE.copy()
    forward_limit = 1.70 if fast_terrain_gait else 1.25
    forward = float(np.clip(command[0], 0.0, forward_limit))
    lateral = float(np.clip(command[1], -0.35, 0.35))
    yaw_rate = float(np.clip(command[2], -1.20, 1.20))
    speed = max(0.0, math.hypot(forward, lateral))
    # Start from the neutral standing pose instead of snapping immediately to
    # a full-stride pose at reset.
    ramp_duration_s = TERRAIN_GAIT_RAMP_S if fast_terrain_gait else 0.80
    ramp_u = float(np.clip(time_s / ramp_duration_s, 0.0, 1.0))
    gait_ramp = ramp_u * ramp_u * (3.0 - 2.0 * ramp_u)
    if fast_terrain_gait:
        cadence_base, cadence_speed_gain, terrain_forward_gain, terrain_forward_limit, terrain_stride_limit, terrain_foot_bias = _terrain_gait_parameters(terrain_task)
        cadence = cadence_base + cadence_speed_gain * speed
    else:
        terrain_forward_gain = 1.25
        terrain_forward_limit = 1.35
        terrain_stride_limit = 0.285
        terrain_foot_bias = 0.0
        cadence = 2.35 + 0.45 * speed
    duty = GAIT_DUTY_FACTOR
    period = 1.0 / cadence
    phase_offsets = (0.0, 0.5, 0.5, 0.0)  # FR, FL, RR, RL diagonal trot
    side_y = (-0.142, 0.142, -0.142, 0.142)
    roll_error = 0.0
    pitch_error = 0.0
    if body_rpy is not None:
        roll_error = float(np.clip(body_rpy[0], -0.45, 0.45))
        pitch_error = float(np.clip(body_rpy[1] - desired_pitch_rad, -0.45, 0.45))
        # Keep the physical course heading with the Go2 IMU.  Without this
        # joint-level yaw reflex, a diagonally loaded foot on uneven ground
        # can slowly turn the base until it walks across (or out of) the
        # course while its body-frame speed still appears correct.
        # The 10% grade creates an early asymmetric stance load.  Correct its
        # yaw before it turns into lateral travel: a yaw error becomes a
        # left/right stride differential through ``forward - yaw_rate*y``.
        yaw_rate_feedback = TERRAIN_YAW_FEEDBACK_GAIN * body_rpy[2]
        if body_angular_velocity is not None:
            yaw_rate_feedback += TERRAIN_YAW_RATE_DAMPING * float(body_angular_velocity[2])
        yaw_rate_feedback += TERRAIN_CROSS_TRACK_YAW_GAIN * float(course_lateral_error_m)
        yaw_rate = float(np.clip(yaw_rate - yaw_rate_feedback, -1.20, 1.20))
    for leg, offset in enumerate(phase_offsets):
        # A modest lead compensates the finite PD/actuator tracking lag.  It
        # changes the foot trajectory, not the root pose, so propulsion still
        # comes from stance contact rather than a hidden kinematic translate.
        local_forward_limit = terrain_forward_limit
        local_forward_gain = terrain_forward_gain
        local_forward = local_forward_gain * float(
            np.clip(forward - yaw_rate * side_y[leg], 0.10, local_forward_limit)
        ) * gait_ramp
        foot_center = terrain_foot_bias
        stride = min(terrain_stride_limit, local_forward * duty * period)
        phase = (cadence * time_s + offset) % 1.0
        # Joint-level IMU reflex: move the physical support polygon under a
        # pitching torso and lengthen the low-side legs under roll.  It uses
        # only Go2's base IMU; no terrain state, root wrench or kinematic base
        # update is injected.
        foot_center += float(np.clip(TERRAIN_PITCH_FOOT_CENTER_GAIN * pitch_error, -0.105, 0.105))
        rear = foot_center - 0.5 * stride
        front = foot_center + 0.5 * stride
        if phase < duty:
            stance_u = phase / duty
            foot_x = front + (rear - front) * stance_u
            lift = 0.0
        else:
            swing_u = (phase - duty) / (1.0 - duty)
            endpoint_slope = -local_forward * (1.0 - duty) * period
            foot_x = _hermite_swing_x(swing_u, rear, front, endpoint_slope)
            # The new level 3 has 80 mm tile-height amplitude (160 mm
            # adjacent peak-to-trough).  This is a real swing-foot clearance,
            # not a terrain bypass or a kinematic root translation.
            # The physical rough surface reaches an 80 mm height amplitude.
            # This clearance is for the swing foot only; stance legs remain
            # entirely contact-driven.
            lift_scale = (
                TERRAIN_SWING_LIFT_BASE_M + TERRAIN_SWING_LIFT_SPEED_GAIN * speed
                if fast_terrain_gait else 0.058 + 0.018 * speed
            )
            lift = lift_scale * math.sin(math.pi * swing_u) ** 2 * gait_ramp
        # A 0.300 m hip-to-foot height settles the loaded official Go2 near
        # its 0.32 m nominal base height instead of the previous 0.21 m crouch.
        side_roll_offset = (-0.065 if leg in (0, 2) else 0.065) * roll_error
        thigh, calf = _sagittal_leg_ik(foot_x, -TERRAIN_STANCE_HEIGHT_M + lift + side_roll_offset)
        index = 3 * leg
        # A front/rear lateral foothold differential is a joint-only yaw
        # moment.  It is kept separate from the velocity-command yaw term so
        # it can correct a grade-induced heading error while all root wrenches
        # remain exactly zero.
        fore_aft_sign = 1.0 if leg in (0, 1) else -1.0
        targets[index] = (
            (0.10 if leg in (0, 2) else -0.10)
            + 0.035 * lateral
            + fore_aft_sign * TERRAIN_YAW_HIP_GAIN * float(body_rpy[2] if body_rpy is not None else 0.0)
        )
        targets[index + 1] = thigh
        targets[index + 2] = calf
    # At reset the official standing pose is already in four-foot contact.
    # Blend the whole joint vector (not just the stride) into the gait so the
    # controller never kicks the robot from its 0.265 m stand height straight
    # to the extended walking pose on the first 5-ms tick.
    return STAND_POSE + gait_ramp * (targets - STAND_POSE)


class Go2LeggedLocoEnv(gym.Env[np.ndarray, np.ndarray]):
    """Velocity tracking with legged-loco-compatible state and action units.

    A phase-conditioned nominal target is a safety prior for a single-robot
    CPU learner.  PPO learns the 12 direct joint residuals around that target;
    this maps a normalized action once by 0.18 rad while avoiding unsafe
    random full-amplitude torques during early optimization.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        seed: int | None = None,
        history_length: int = 9,
        max_steps: int = 750,
        domain_randomization: bool = True,
        sensor_noise: bool = True,
        terrain_task: str = "flat",
        rough_level: int | None = None,
    ) -> None:
        super().__init__()
        if history_length < 0:
            raise ValueError("history_length must be non-negative")
        self.terrain_task = validate_terrain_task(terrain_task)
        if self.terrain_task != "rough" and rough_level is not None:
            raise ValueError("rough_level is only valid for terrain_task='rough'")
        self._requested_rough_level = validate_rough_level(rough_level) if rough_level is not None else None
        self._active_rough_level = self._requested_rough_level or 2
        self.model = mujoco.MjModel.from_xml_string(
            build_go2_landing_xml(include_drone=False, terrain_task=self.terrain_task)
        )
        self.data = mujoco.MjData(self.model)
        self.base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.joint_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_NAMES], dtype=np.int32
        )
        self.qposadr = self.model.jnt_qposadr[self.joint_ids].astype(np.int32)
        self.dofadr = self.model.jnt_dofadr[self.joint_ids].astype(np.int32)
        self.actuator_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in ACTUATOR_NAMES], dtype=np.int32
        )
        self.foot_geom_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in LEG_ORDER], dtype=np.int32
        )
        self.ground_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        self.terrain_geom_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in terrain_geom_names(self.terrain_task)],
            dtype=np.int32,
        )
        if np.any(self.terrain_geom_ids < 0):
            raise RuntimeError("terrain geometry was not compiled into the locomotion scene")
        self._randomized_friction_geom_ids = np.concatenate(
            (np.array([self.ground_geom_id], dtype=np.int32), self.terrain_geom_ids, self.foot_geom_ids)
        )
        self._nominal_friction = self.model.geom_friction[self._randomized_friction_geom_ids].copy()
        self.physics_steps = 4  # legged-loco: 5 ms physics, decimation=4
        self.dt = self.physics_steps * float(self.model.opt.timestep)
        self.history_length = history_length
        self.single_observation_dim = 45  # 3 omega + 3 rpy + 3 command + 12 q + 12 dq + 12 previous action
        # The upstream history wrapper concatenates the current policy state
        # with nine proprioception frames.  Both groups have the same 45 terms.
        observation_dim = self.single_observation_dim * (history_length + 1)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(observation_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        self.max_steps = max_steps
        self.domain_randomization = domain_randomization
        self.sensor_noise = sensor_noise
        self._history: deque[np.ndarray] = deque(maxlen=history_length)
        # Keep enough history to randomize 3--5 command-step latency in
        # training while the integrated adapter uses the nominal four steps.
        self._delayed_actions: deque[np.ndarray] = deque(maxlen=7)
        self._last_action = np.zeros(12, dtype=np.float64)
        self._command = np.zeros(3, dtype=np.float64)
        self._step_count = 0
        self._path_length = 0.0
        self._previous_position = np.zeros(3, dtype=np.float64)
        self._previous_foot_positions = np.zeros((4, 3), dtype=np.float64)
        self._previous_foot_contact_mask = np.zeros(4, dtype=bool)
        self._last_stance_slip_mps = 0.0
        self._step_torque_saturation_fraction = 0.0
        self._step_root_wrench_max_abs = 0.0
        self._motor_strength_scale = np.ones(12, dtype=np.float64)
        self._kp_scale = 1.0
        self._kd_scale = 1.0
        self._delay_steps = 4
        self._command_change_at = 0
        self.reset(seed=seed)

    @property
    def base_position(self) -> np.ndarray:
        return self.data.xpos[self.base_id].copy()

    @property
    def base_velocity(self) -> np.ndarray:
        # Source velocity commands are expressed in the base frame.
        rotation = self.data.xmat[self.base_id].reshape(3, 3)
        return rotation.T @ self.data.qvel[:3]

    def _sample_command(self) -> None:
        if self.terrain_task != "flat":
            # The recordings follow a fixed forward route.  Preserve its
            # exactly three-times faster nominal command while excluding
            # unrelated sideways/yaw random commands that would turn a
            # foothold test into a course-boundary test.
            if self.terrain_task == "rough":
                self._command[:] = (
                    0.70 * 0.48 * TERRAIN_SPEED_MULTIPLIER,
                    0.0,
                    0.0,
                )
                return
            self._command[:] = (
                0.70 * 0.52 * TERRAIN_SPEED_MULTIPLIER,
                0.0,
                0.0,
            )
            return
        self._command[:] = (
            self.np_random.uniform(0.35, 1.20),
            self.np_random.uniform(-0.42, 0.42),
            self.np_random.uniform(-0.65, 0.65),
        )

    def _gait_target(self, time_s: float) -> np.ndarray:
        """Stable trot prior; the learned action supplies 12 source-scale offsets."""
        return legged_loco_reference_target(
            time_s,
            self._command,
            fast_terrain_gait=self.terrain_task != "flat",
            terrain_task=self.terrain_task,
            body_rpy=_rpy(self.data.xmat[self.base_id]) if self.terrain_task != "flat" else None,
            body_angular_velocity=self.data.cvel[self.base_id, :3] if self.terrain_task != "flat" else None,
            course_lateral_error_m=float(self.base_position[1]) if self.terrain_task != "flat" else 0.0,
            desired_pitch_rad=terrain_initial_pitch_rad(self.terrain_task),
        )

    def _apply_control(self, action: np.ndarray, *, update_delay: bool) -> None:
        qpos = self.data.qpos[self.qposadr]
        qvel = self.data.qvel[self.dofadr]
        conditioned_action = np.asarray(action, dtype=np.float64).clip(-1.0, 1.0)
        if self.terrain_task != "flat":
            conditioned_action = TERRAIN_RESIDUAL_POLICY_GAIN * conditioned_action
        # Four 20 ms command steps of latency from legged-loco's DelayedPDActuatorCfg.
        if update_delay:
            self._delayed_actions.append(conditioned_action)
        delayed = list(self._delayed_actions)[-(self._delay_steps + 1)]
        time_s = float(self.data.time)
        reference = self._gait_target(time_s)
        next_reference = self._gait_target(time_s + float(self.model.opt.timestep))
        targets = reference + JOINT_RESIDUAL_SCALE_RAD * delayed
        target_velocity = (next_reference - reference) / float(self.model.opt.timestep)
        torque = self._motor_strength_scale * (
            60.0 * self._kp_scale * (targets - qpos)
            + 2.0 * self._kd_scale * (target_velocity - qvel)
        )
        ctrlrange = self.model.actuator_ctrlrange[self.actuator_ids]
        self.data.ctrl[self.actuator_ids] = np.clip(torque, ctrlrange[:, 0], ctrlrange[:, 1])
        saturated = (torque <= ctrlrange[:, 0]) | (torque >= ctrlrange[:, 1])
        self._step_torque_saturation_fraction = max(
            self._step_torque_saturation_fraction, float(np.mean(saturated))
        )

        # No hidden root force or body torque: motion and balance must be
        # generated solely by 12 joint actuators and physical foot contact.
        self.data.xfrc_applied[self.base_id] = 0.0
        self._step_root_wrench_max_abs = max(
            self._step_root_wrench_max_abs,
            float(np.max(np.abs(self.data.xfrc_applied[self.base_id, :6]))),
        )

    def _single_observation(self) -> np.ndarray:
        angular_velocity = np.clip(self.data.cvel[self.base_id, :3], -8.0, 8.0)
        orientation = np.clip(_rpy(self.data.xmat[self.base_id]), -math.pi, math.pi)
        joint_position = np.clip(self.data.qpos[self.qposadr] - STAND_POSE, -2.0, 2.0)
        joint_velocity = np.clip(self.data.qvel[self.dofadr], -20.0, 20.0)
        if self.sensor_noise:
            # Go2's policy deploys from its IMU and 12 joint encoders.  These
            # small seeded perturbations train against their finite accuracy.
            angular_velocity = angular_velocity + self.np_random.normal(0.0, 0.006, size=3)
            orientation = orientation + self.np_random.normal(0.0, 0.0025, size=3)
            joint_position = joint_position + self.np_random.normal(0.0, 0.0015, size=12)
            joint_velocity = joint_velocity + self.np_random.normal(0.0, 0.018, size=12)
        return np.concatenate(
            (
                angular_velocity,
                orientation,
                self._command,
                joint_position,
                joint_velocity,
                self._last_action,
            )
        )

    def _observation(self, current: np.ndarray | None = None) -> np.ndarray:
        # Sample the physical sensors once per control tick.  In particular,
        # do not create two independent noise draws at the same timestamp for
        # the current term and newest history term.
        if current is None:
            current = self._single_observation()
        if self.history_length == 0:
            return current.astype(np.float32)
        history = list(self._history)
        if not history:
            history = [current.copy()] * self.history_length
        while len(history) < self.history_length:
            history.insert(0, history[0].copy())
        return np.concatenate((current, *history[-self.history_length:])).astype(np.float32)

    def _foot_contact_mask(self) -> np.ndarray:
        contacts = np.zeros(4, dtype=bool)
        support_geoms = {int(self.ground_geom_id), *(int(geom_id) for geom_id in self.terrain_geom_ids)}
        for index in range(self.data.ncon):
            pair = {int(self.data.contact[index].geom1), int(self.data.contact[index].geom2)}
            if not pair.intersection(support_geoms):
                continue
            for foot_index, foot in enumerate(self.foot_geom_ids):
                if int(foot) in pair:
                    contacts[foot_index] = True
        return contacts

    def _expected_contact_mask(self) -> np.ndarray:
        speed = float(np.linalg.norm(self._command[:2]))
        cadence = (
            _terrain_gait_parameters(self.terrain_task)[0]
            + _terrain_gait_parameters(self.terrain_task)[1] * speed
            if self.terrain_task != "flat" else 2.35 + 0.45 * speed
        )
        duty = GAIT_DUTY_FACTOR
        offsets = np.array((0.0, 0.5, 0.5, 0.0), dtype=np.float64)
        phase = (cadence * float(self.data.time) + offsets) % 1.0
        return phase < duty

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, float]]:
        super().reset(seed=seed)
        if self.terrain_task == "rough":
            self._active_rough_level = (
                self._requested_rough_level
                if self._requested_rough_level is not None
                else int(self.np_random.integers(1, 4))
            )
            configure_rough_terrain(self.model, level=self._active_rough_level)
        mujoco.mj_resetData(self.model, self.data)
        initial_ground_height = terrain_height_at(
            self.terrain_task, 0.0, 0.0,
            rough_level=self._active_rough_level if self.terrain_task == "rough" else None,
        )
        half_pitch = 0.5 * terrain_initial_pitch_rad(self.terrain_task)
        self.data.qpos[:7] = (
            0.0, 0.0, initial_ground_height + 0.32,
            math.cos(half_pitch), 0.0, math.sin(half_pitch), 0.0,
        )
        # On the stepped terrain, begin from a contact-safe nominal posture
        # rather than injecting a 12 mrad joint kick that can put a foot
        # inside a neighbouring 80 mm tile before the first control tick.
        # Runtime robustness is still trained by friction, motor, PD, delay,
        # IMU and encoder randomization below.
        initial_joint_noise = 0.0 if self.terrain_task != "flat" else 0.012
        self.data.qpos[self.qposadr] = STAND_POSE + self.np_random.normal(0.0, initial_joint_noise, size=12)
        self.data.qvel[:] = 0.0
        if self.domain_randomization:
            terrain_scale = 0.02 if self.terrain_task != "flat" else 0.12
            friction_scale = self.np_random.uniform(
                1.0 - terrain_scale, 1.0 + terrain_scale,
                size=(len(self._randomized_friction_geom_ids), 1),
            )
            self.model.geom_friction[self._randomized_friction_geom_ids] = self._nominal_friction * friction_scale
            motor_scale = 0.02 if self.terrain_task != "flat" else 0.06
            self._motor_strength_scale[:] = self.np_random.uniform(1.0 - motor_scale, 1.0 + motor_scale, size=12)
            self._kp_scale = float(self.np_random.uniform(1.0 - motor_scale, 1.0 + motor_scale))
            self._kd_scale = float(self.np_random.uniform(1.0 - motor_scale, 1.0 + motor_scale))
            # Terrain verification keeps the task's documented nominal four
            # low-level ticks.  The broad 3--5 tick latency curriculum is
            # retained for flat deployment, where it is not coupled to a
            # 10% grade and 160 mm peak-to-trough height transition.
            self._delay_steps = 4 if self.terrain_task != "flat" else int(self.np_random.integers(3, 6))
        else:
            self.model.geom_friction[self._randomized_friction_geom_ids] = self._nominal_friction
            self._motor_strength_scale[:] = 1.0
            self._kp_scale = 1.0
            self._kd_scale = 1.0
            self._delay_steps = 4
        self._sample_command()
        self._last_action[:] = 0.0
        self._history.clear()
        self._delayed_actions.clear()
        for _ in range(self._delayed_actions.maxlen or 7):
            self._delayed_actions.append(np.zeros(12, dtype=np.float64))
        self._step_count = 0
        self._path_length = 0.0
        self._command_change_at = int(self.np_random.integers(150, 350))
        mujoco.mj_forward(self.model, self.data)
        self._previous_position = self.base_position
        self._previous_foot_positions = self.data.geom_xpos[self.foot_geom_ids].copy()
        self._previous_foot_contact_mask = self._foot_contact_mask()
        self._last_stance_slip_mps = 0.0
        self._step_torque_saturation_fraction = 0.0
        self._step_root_wrench_max_abs = 0.0
        current = self._single_observation()
        for _ in range(self.history_length):
            self._history.append(current.copy())
        return self._observation(current), {
            "command_x_mps": float(self._command[0]),
            "payload_kg": 0.22,
            "terrain_task": self.terrain_task,
            "rough_level": float(self._active_rough_level) if self.terrain_task == "rough" else 0.0,
        }

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, float]]:
        action = np.asarray(action, dtype=np.float64).clip(-1.0, 1.0)
        self._step_count += 1
        if self._step_count >= self._command_change_at:
            self._sample_command()
            self._command_change_at += int(self.np_random.integers(150, 350))
        stable_slip_samples: list[float] = []
        self._step_torque_saturation_fraction = 0.0
        self._step_root_wrench_max_abs = 0.0
        for physics_index in range(self.physics_steps):
            self._apply_control(action, update_delay=physics_index == 0)
            mujoco.mj_step(self.model, self.data)
            foot_positions = self.data.geom_xpos[self.foot_geom_ids].copy()
            foot_velocity = (foot_positions - self._previous_foot_positions) / float(self.model.opt.timestep)
            contact_mask = self._foot_contact_mask()
            stable_contact = contact_mask & self._previous_foot_contact_mask
            stable_slip_samples.extend(np.linalg.norm(foot_velocity[stable_contact, :2], axis=1).tolist())
            self._previous_foot_positions = foot_positions
            self._previous_foot_contact_mask = contact_mask
        self._last_stance_slip_mps = float(np.mean(stable_slip_samples)) if stable_slip_samples else 0.0
        velocity = self.base_velocity
        # The documented source task tracks base-frame velocity on flat
        # ground.  Terrain certification additionally requires the robot to
        # make progress along the visible world-X course; otherwise a yawed
        # robot can score well while walking back down an uphill slope.
        if self.terrain_task != "flat":
            world_velocity = self.data.qvel[:3]
            velocity_error = np.array(
                [self._command[0] - world_velocity[0], -world_velocity[1]],
                dtype=np.float64,
            )
        else:
            velocity_error = self._command[:2] - velocity[:2]
        yaw_rate_error = float(self._command[2] - self.data.cvel[self.base_id, 2])
        base_up = float(self.data.xmat[self.base_id, 8])
        position = self.base_position
        self._path_length += float(np.linalg.norm(position[:2] - self._previous_position[:2]))
        self._previous_position = position
        contact_mask = self._foot_contact_mask()
        feet = int(np.count_nonzero(contact_mask))
        gait_match = float(np.mean(contact_mask == self._expected_contact_mask()))
        terrain_height = terrain_height_at(
            self.terrain_task,
            float(position[0]),
            float(position[1]),
            rough_level=self._active_rough_level if self.terrain_task == "rough" else None,
        )
        height_error = float(position[2] - (terrain_height + 0.32))
        assist_force = float(np.linalg.norm(self.data.xfrc_applied[self.base_id, :2]))
        tilt_deg = math.degrees(math.acos(float(np.clip(base_up, -1.0, 1.0))))
        body_rpy = _rpy(self.data.xmat[self.base_id])
        terrain_attitude_error = float(math.hypot(
            body_rpy[0], body_rpy[1] - terrain_initial_pitch_rad(self.terrain_task),
        ))
        course_heading_error = float(math.atan2(math.sin(body_rpy[2]), math.cos(body_rpy[2])))
        effective_action = action * (TERRAIN_RESIDUAL_POLICY_GAIN if self.terrain_task != "flat" else 1.0)
        if self.terrain_task != "flat":
            # A fall is far costlier than a momentary speed error.  This is a
            # real joint-contact task: no terrain height enters the policy,
            # only the reward/evaluation label.  Reward a broad physical
            # support polygon and surface-relative IMU attitude so the policy
            # learns to plant and recover before it accelerates.
            reward = (
                4.0 * math.exp(-3.2 * float(np.dot(velocity_error, velocity_error)))
                + 1.2 * math.exp(-2.4 * yaw_rate_error * yaw_rate_error)
                + 2.4 * math.exp(-16.0 * terrain_attitude_error * terrain_attitude_error)
                + 1.4 * math.exp(-4.0 * course_heading_error * course_heading_error)
                + 0.90 * min(feet, 3) / 3.0
                + 0.30 * gait_match
                - 8.0 * height_error * height_error
                - 0.55 * min(self._last_stance_slip_mps, 2.0)
                - 0.015 * float(np.dot(effective_action, effective_action))
                - 0.035 * float(np.dot(effective_action - self._last_action, effective_action - self._last_action))
                - 0.16 * self._step_torque_saturation_fraction
            )
        else:
            reward = (
                2.5 * math.exp(-3.0 * float(np.dot(velocity_error, velocity_error)))
                + 2.0 * math.exp(-2.4 * yaw_rate_error * yaw_rate_error)
                + 0.40 * max(0.0, base_up)
                + 0.25 * gait_match
                - 5.0 * height_error * height_error
                - 0.40 * min(self._last_stance_slip_mps, 2.0)
                - 0.10 * float(np.dot(action, action))
                - 0.12 * float(np.dot(action - self._last_action, action - self._last_action))
                - 0.20 * self._step_torque_saturation_fraction
            )
        fallen = position[2] < 0.18 or base_up < 0.55
        if fallen:
            reward -= 260.0 if self.terrain_task != "flat" else 40.0
        self._last_action = effective_action
        current_observation = self._single_observation()
        observation = self._observation(current_observation)
        self._history.append(current_observation)
        route_complete = bool(
            self.terrain_task != "flat" and position[0] >= TERRAIN_ROUTE_TARGET_X_M
        )
        if route_complete and not fallen:
            # Finish before the visible slope/heightfield boundary rather than
            # rewarding a controller for walking off a finite collision geom.
            reward += 60.0
        terminated = fallen
        truncated = self._step_count >= self.max_steps or route_complete
        info = {
            "command_x_mps": float(self._command[0]),
            "command_y_mps": float(self._command[1]),
            "actual_x_mps": float(velocity[0]),
            "actual_y_mps": float(velocity[1]),
            "world_x_mps": float(self.data.qvel[0]),
            "world_y_mps": float(self.data.qvel[1]),
            "world_x_m": float(position[0]),
            "world_y_m": float(position[1]),
            "velocity_error_mps": float(np.linalg.norm(velocity_error)),
            "yaw_rate_error_radps": abs(yaw_rate_error),
            "base_up": base_up,
            "foot_contacts": float(feet),
            "gait_contact_match": gait_match,
            "stance_foot_slip_mps": self._last_stance_slip_mps,
            "base_height_m": float(position[2]),
            "terrain_ground_height_m": terrain_height,
            "terrain_rough_level": float(self._active_rough_level) if self.terrain_task == "rough" else 0.0,
            "assist_force_n": assist_force,
            "root_wrench_max_abs": self._step_root_wrench_max_abs,
            "torque_saturation_fraction": self._step_torque_saturation_fraction,
            "base_tilt_deg": tilt_deg,
            "surface_relative_tilt_deg": math.degrees(terrain_attitude_error),
            "course_heading_error_deg": math.degrees(abs(course_heading_error)),
            "route_complete": float(route_complete),
            "path_distance_m": self._path_length,
            "fall": float(fallen),
            "episode_steps": float(self._step_count),
        }
        return observation, float(reward), terminated, truncated, info

    def close(self) -> None:
        return None
