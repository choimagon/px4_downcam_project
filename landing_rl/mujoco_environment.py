"""MuJoCo physics environment for moving-QR precision landing.

Unlike the earlier analytical training environment, every state transition in
this module advances a MuJoCo free-body drone.  The policy retains the same
six-element vision/control observation used by the deployed controller, while
the pad, drone inertia, gravity, drag-like velocity servo and landing contact
all live in MuJoCo.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .scenario import INNER_RING_RADIUS_M, MOTION_DIFFICULTIES, OUTER_RING_RADIUS_M, WavyMotionProfile, random_motion_profile


ASSET_DIR = Path(__file__).resolve().parents[1] / "assets" / "mujoco_x500"


MUJOCO_XML = rf"""
<mujoco model="moving_qr_precision_landing">
  <compiler angle="radian" coordinate="local" meshdir="{ASSET_DIR.as_posix()}"/>
  <option timestep="0.01" gravity="0 0 -9.81" integrator="RK4"/>
  <visual><global offwidth="1280" offheight="720"/><map znear="0.001" zfar="100"/><rgba haze="0.16 0.23 0.34 1"/></visual>
  <asset>
    <texture name="ground" type="2d" builtin="checker" rgb1="0.10 0.16 0.20" rgb2="0.14 0.21 0.27" width="512" height="512" mark="none"/>
    <material name="ground" texture="ground" texrepeat="8 8" reflectance="0.15"/>
    <material name="drone" rgba="0.08 0.45 0.92 1"/><material name="arm" rgba="0.05 0.10 0.16 1"/>
    <material name="x500_metal" rgba="0.10 0.13 0.17 1"/><material name="camera_housing" rgba="0.035 0.045 0.060 1"/><material name="lens" rgba="0.15 0.35 0.46 1"/>
    <material name="qrwhite" rgba="0.96 0.96 0.94 1"/><material name="qrblack" rgba="0.015 0.015 0.018 1"/>
    <material name="ring2" rgba="0.18 0.85 0.95 0.16"/><material name="ring7" rgba="0.98 0.63 0.15 0.12"/>
    <!-- These are converted directly from the PX4 Gazebo x500_base SDF
         assets.  MuJoCo does not ingest Gazebo SDF/Collada natively. -->
    <mesh name="x500_frame" file="x500_frame.obj"/>
    <mesh name="x500_motor" file="x500_motor.obj"/>
    <mesh name="x500_bell" file="x500_bell.obj"/>
    <mesh name="x500_prop_cw" file="1345_prop_cw.stl" scale="0.846154 0.846154 0.846154"/>
    <mesh name="x500_prop_ccw" file="1345_prop_ccw.stl" scale="0.846154 0.846154 0.846154"/>
  </asset>
  <worldbody>
    <light name="key" pos="-3 -4 8" dir="0.3 0.4 -1" directional="true" diffuse="0.9 0.95 1"/>
    <light name="fill" pos="5 4 4" dir="-1 -1 -0.5" diffuse="0.35 0.45 0.65"/>
    <geom name="ground" type="plane" size="32 32 .1" material="ground"/>
    <!-- Subtle translucent landing-search zones.  Filled geometry avoids the
         coplanar depth artefacts that nested disk cut-outs cause in video. -->
    <geom name="outer_search_zone" type="cylinder" size="7.0 .004" pos="0 0 .006" material="ring7" contype="0" conaffinity="0"/>
    <geom name="inner_safe_zone" type="cylinder" size="2.0 .006" pos="0 0 .012" material="ring2" contype="0" conaffinity="0"/>
    <body name="qr_pad" mocap="true" pos="0 0 .04">
      <geom name="pad_base" type="box" size=".245 .245 .018" rgba="0.18 0.22 0.28 1" contype="0" conaffinity="0"/>
      <geom name="qr_white" type="box" pos="0 0 .021" size=".20 .20 .004" material="qrwhite" contype="0" conaffinity="0"/>
      <geom type="box" pos="-.13 -.13 .027" size=".045 .045 .003" material="qrblack" contype="0" conaffinity="0"/>
      <geom type="box" pos=" .13 -.13 .027" size=".045 .045 .003" material="qrblack" contype="0" conaffinity="0"/>
      <geom type="box" pos="-.13  .13 .027" size=".045 .045 .003" material="qrblack" contype="0" conaffinity="0"/>
      <geom type="box" pos=" .02  .01 .027" size=".030 .030 .003" material="qrblack" contype="0" conaffinity="0"/>
      <geom type="box" pos=" .10  .06 .027" size=".024 .024 .003" material="qrblack" contype="0" conaffinity="0"/>
      <geom type="box" pos="-.04 -.04 .027" size=".020 .020 .003" material="qrblack" contype="0" conaffinity="0"/>
      <geom type="box" pos=" .00  .12 .027" size=".018 .018 .003" material="qrblack" contype="0" conaffinity="0"/>
    </body>
    <body name="drone" pos="0 0 1.5">
      <freejoint/>
      <!-- x500_base/base_link mass and diagonal inertia from the original
           Gazebo SDF.  Collision remains deliberately simple for stable RL,
           while the rendered vehicle is the converted original X500 mesh. -->
      <inertial pos="0 0 0" mass="2.0" diaginertia="0.021666667 0.021666667 0.040000000"/>
      <!-- Group 1 is the vehicle's own geometry.  The attached down camera
           hides this group only in its optical render, so propellers and
           landing hardware never obscure the QR image. -->
      <geom name="drone_collision" type="box" pos="0 0 .007" size=".1768 .1768 .025" rgba="0 0 0 0" group="1"/>
      <geom name="x500_frame_visual" type="mesh" mesh="x500_frame" pos="0 0 .025" euler="0 0 3.141593" group="1" contype="0" conaffinity="0"/>
      <geom type="mesh" mesh="x500_motor" pos=" .174  .174 .032" euler="0 0 -.45" material="x500_metal" group="1" contype="0" conaffinity="0"/>
      <geom type="mesh" mesh="x500_motor" pos="-.174  .174 .032" euler="0 0 -.45" material="x500_metal" group="1" contype="0" conaffinity="0"/>
      <geom type="mesh" mesh="x500_motor" pos=" .174 -.174 .032" euler="0 0 -.45" material="x500_metal" group="1" contype="0" conaffinity="0"/>
      <geom type="mesh" mesh="x500_motor" pos="-.174 -.174 .032" euler="0 0 -.45" material="x500_metal" group="1" contype="0" conaffinity="0"/>
      <geom type="mesh" mesh="x500_bell" pos=" .174 -.174 .028" material="x500_metal" group="1" contype="0" conaffinity="0"/>
      <geom type="mesh" mesh="x500_bell" pos="-.174  .174 .028" material="x500_metal" group="1" contype="0" conaffinity="0"/>
      <geom type="mesh" mesh="x500_bell" pos=" .174  .174 .028" material="x500_metal" group="1" contype="0" conaffinity="0"/>
      <geom type="mesh" mesh="x500_bell" pos="-.174 -.174 .028" material="x500_metal" group="1" contype="0" conaffinity="0"/>
      <!-- PX4 Gazebo's prop meshes have a non-zero local visual pose beneath
           each rotor link.  Keeping that offset aligns the rendered blades
           with the motor axes after MuJoCo imports the STL mesh frame. -->
      <geom name="propeller_front_right" type="mesh" mesh="x500_prop_ccw" pos=" .152 -.320384615 .044" material="arm" group="1" contype="0" conaffinity="0"/>
      <geom name="propeller_rear_left" type="mesh" mesh="x500_prop_ccw" pos="-.196  .027615385 .044" material="arm" group="1" contype="0" conaffinity="0"/>
      <geom name="propeller_front_left" type="mesh" mesh="x500_prop_cw" pos=" .152052  .026055 .044" material="arm" group="1" contype="0" conaffinity="0"/>
      <geom name="propeller_rear_right" type="mesh" mesh="x500_prop_cw" pos="-.195948 -.321945 .044" material="arm" group="1" contype="0" conaffinity="0"/>
      <!-- Original mono_cam SDF housing dimensions (2 × 4 × 4 cm) and its
           downward-facing placement on the X500. -->
      <geom name="mono_cam_housing" type="box" pos="0 0 -.100" size=".010 .020 .020" material="camera_housing" group="1" contype="0" conaffinity="0"/>
      <geom name="mono_cam_lens" type="cylinder" pos="0 0 -.123" size=".009 .006" material="lens" group="1" contype="0" conaffinity="0"/>
      <!-- MuJoCo's camera forward axis is already local -Z, matching the
           downward-facing Gazebo mono_cam_down mounting on this Z-up body. -->
      <camera name="down_camera" pos="0 0 -.045" fovy="145"/>
    </body>
  </worldbody>
</mujoco>
"""


MUJOCO_PROFILES: dict[str, dict[str, float | tuple[float, float]]] = {
    # Training starts from the requested annulus.  A deterministic broad
    # acquisition controller approaches the moving landing zone until it is in
    # the down-camera FOV; the RL action only owns safe final centering.
    "train": {"max_steps": 520, "altitude": (1.35, 1.75), "radius": (2.01, 6.99), "max_speed": 1.10, "max_descent": 0.22, "wind": 0.030, "dropout": 0.008, "alignment": 0.12, "landing": 0.15},
    "easy": {"max_steps": 650, "altitude": (1.35, 1.90), "radius": (2.01, 4.00), "max_speed": 1.15, "max_descent": 0.20, "wind": 0.035, "dropout": 0.010, "alignment": 0.12, "landing": 0.15},
    "medium": {"max_steps": 800, "altitude": (1.45, 2.20), "radius": (3.00, 5.50), "max_speed": 1.25, "max_descent": 0.18, "wind": 0.045, "dropout": 0.014, "alignment": 0.13, "landing": 0.16},
    "hard": {"max_steps": 1_000, "altitude": (1.55, 2.50), "radius": (4.50, OUTER_RING_RADIUS_M - 0.01), "max_speed": 1.38, "max_descent": 0.16, "wind": 0.060, "dropout": 0.020, "alignment": 0.14, "landing": 0.17},
}


class MujocoQrPrecisionLandingEnv(gym.Env[np.ndarray, np.ndarray]):
    """Moving QR landing environment whose step function advances MuJoCo."""

    metadata = {"render_modes": []}

    def __init__(self, *, difficulty: str = "train", seed: int | None = None, dt: float = 0.10) -> None:
        super().__init__()
        if difficulty not in MOTION_DIFFICULTIES:
            raise ValueError(f"difficulty must be one of: {', '.join(MOTION_DIFFICULTIES)}")
        self.difficulty = difficulty
        self.profile = MUJOCO_PROFILES[difficulty]
        self.model = mujoco.MjModel.from_xml_string(MUJOCO_XML)
        self.data = mujoco.MjData(self.model)
        self.drone_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "drone")
        self.pad_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "qr_pad")
        self.pad_mocap_id = int(self.model.body_mocapid[self.pad_id])
        self.mass = float(self.model.body_mass[self.drone_id])
        self.physics_steps = max(1, round(dt / self.model.opt.timestep))
        self.dt = self.physics_steps * float(self.model.opt.timestep)
        self.max_steps = int(self.profile["max_steps"])
        self.max_speed = float(self.profile["max_speed"])
        self.max_descent = float(self.profile["max_descent"])
        self.alignment_error = float(self.profile["alignment"])
        self.landing_error = float(self.profile["landing"])
        self.observation_space = spaces.Box(
            low=np.array([-1, -1, 0, 0, -1, -1], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self._motion: WavyMotionProfile | None = None
        self._motion_time = 0.0
        self._step_count = 0
        self._aligned_streak = 0
        self._dropout = False
        self._pad_position = np.zeros(3, dtype=np.float64)
        self._pad_velocity = np.zeros(2, dtype=np.float64)
        self.reset(seed=seed)

    @property
    def drone_position(self) -> np.ndarray:
        return self.data.xpos[self.drone_id].copy()

    @property
    def drone_velocity(self) -> np.ndarray:
        return self.data.qvel[:3].copy()

    @property
    def pad_position(self) -> np.ndarray:
        return self._pad_position.copy()

    @property
    def target_velocity(self) -> np.ndarray:
        return self._pad_velocity.copy()

    def _set_pad_pose(self, motion_time: float) -> None:
        assert self._motion is not None
        x, y = self._motion.position_at(motion_time)
        roll, pitch, yaw, heave = self._motion.wave_at(motion_time)
        self._pad_position[:] = (x, y, 0.04 + heave)
        self._pad_velocity[:] = self._motion.velocity_at(motion_time)
        self.data.mocap_pos[self.pad_mocap_id] = self._pad_position
        # Small deck motion is visible to the renderer and perturbs the QR
        # center.  MuJoCo quaternions use w, x, y, z.
        cr, sr = math.cos(math.radians(roll) / 2), math.sin(math.radians(roll) / 2)
        cp, sp = math.cos(math.radians(pitch) / 2), math.sin(math.radians(pitch) / 2)
        cy, sy = math.cos(math.radians(yaw) / 2), math.sin(math.radians(yaw) / 2)
        self.data.mocap_quat[self.pad_mocap_id] = (cy * cp * cr + sy * sp * sr, cy * cp * sr - sy * sp * cr, cy * sp * cr + sy * cp * sr, sy * cp * cr - cy * sp * sr)

    def _relative_altitude(self) -> float:
        return max(0.0, float(self.drone_position[2] - (self._pad_position[2] + 0.025)))

    def _horizontal_error(self) -> np.ndarray:
        return self.drone_position[:2] - self._pad_position[:2]

    def _detected(self, horizontal_distance: float) -> bool:
        # Approximate a 76-degree downward-camera footprint and retain small
        # detector dropouts.  This is a visual visibility model, not a target
        # coordinate fed directly to the learned policy.
        footprint = max(1.15, self._relative_altitude() * 1.55)
        return horizontal_distance <= footprint and not self._dropout

    def _observation(self) -> np.ndarray:
        error = self._horizontal_error()
        horizontal_distance = float(np.linalg.norm(error))
        detected = self._detected(horizontal_distance)
        if detected:
            # Target image center relative to drone image center, normalized
            # to the usable QR camera field of view.
            camera_error = np.clip(-error / max(1.20, self._relative_altitude() * 1.55), -1.0, 1.0)
        else:
            camera_error = np.zeros(2, dtype=np.float64)
        return np.array(
            [camera_error[0], camera_error[1], min(1.0, self._relative_altitude() / 2.5), float(detected), self._pad_velocity[0] / 1.4, self._pad_velocity[1] / 1.4],
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, float]]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        radius_low, radius_high = self.profile["radius"]  # type: ignore[misc]
        altitude_low, altitude_high = self.profile["altitude"]  # type: ignore[misc]
        radius = float(self.np_random.uniform(radius_low, radius_high))
        heading = float(self.np_random.uniform(-math.pi, math.pi))
        altitude = float(self.np_random.uniform(altitude_low, altitude_high))
        self.data.qpos[:7] = (radius * math.cos(heading), radius * math.sin(heading), altitude, 1.0, 0.0, 0.0, 0.0)
        self.data.qvel[:] = 0.0
        self._motion = random_motion_profile(int(self.np_random.integers(1, 2**31 - 1)), self.difficulty)
        self._motion_time = 0.0
        self._step_count = 0
        self._aligned_streak = 0
        self._dropout = False
        self._set_pad_pose(0.0)
        mujoco.mj_forward(self.model, self.data)
        return self._observation(), {"radius_m": radius, "altitude_m": self._relative_altitude(), "trajectory_seed": float(self._motion.seed)}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, float]]:
        self._step_count += 1
        action = np.asarray(action, dtype=np.float64).clip(-1.0, 1.0)
        previous_distance = float(np.linalg.norm(self._horizontal_error()))
        observation = self._observation()
        detected = bool(observation[3] > 0.5)
        relative = self._horizontal_error()
        current_velocity = self.drone_velocity

        if detected:
            visual_servo = -relative / max(1.20, self._relative_altitude() * 1.55)
            # The learned action is a deliberately small residual.  It can
            # refine a visual-servo solution but cannot overturn a safely
            # centered approach when an off-policy actor briefly saturates.
            desired_xy = self._pad_velocity + 0.92 * visual_servo * self.max_speed + 0.008 * action * self.max_speed
        else:
            # Broad acquisition remains deterministic and only moves toward a
            # landing-zone prior.  Once the QR is visible, the policy's action
            # governs final centering and descent.
            desired_xy = self._pad_velocity - 0.46 * relative
            speed = float(np.linalg.norm(desired_xy))
            if speed > 1.50:
                desired_xy *= 1.50 / speed

        relative_velocity = current_velocity[:2] - self._pad_velocity
        aligned = detected and float(np.linalg.norm(relative)) < self.alignment_error and float(np.linalg.norm(relative_velocity)) < 0.26
        self._aligned_streak = self._aligned_streak + 1 if aligned else 0
        descent = self.max_descent if self._aligned_streak >= 5 else 0.0
        desired_velocity = np.array([desired_xy[0], desired_xy[1], -descent], dtype=np.float64)
        wind_sigma = float(self.profile["wind"])

        for physics_index in range(self.physics_steps):
            self._motion_time += float(self.model.opt.timestep)
            self._set_pad_pose(self._motion_time)
            velocity_error = desired_velocity - self.drone_velocity
            acceleration = np.array([4.6 * velocity_error[0], 4.6 * velocity_error[1], 5.8 * velocity_error[2]], dtype=np.float64)
            acceleration[:2] += self.np_random.normal(0.0, wind_sigma, size=2)
            force = self.mass * (acceleration - self.model.opt.gravity)
            self.data.xfrc_applied[self.drone_id] = 0.0
            # x500_base is 2.0 kg in the original Gazebo SDF, so hover alone
            # requires 19.62 N.  Keep headroom above that physical value.
            self.data.xfrc_applied[self.drone_id, :3] = np.clip(force, -32.0, 32.0)
            mujoco.mj_step(self.model, self.data)

        self._dropout = bool(self.np_random.random() < float(self.profile["dropout"]))
        horizontal_distance = float(np.linalg.norm(self._horizontal_error()))
        relative_altitude = self._relative_altitude()
        success = relative_altitude <= 0.15 and horizontal_distance < self.landing_error
        hard_landing = relative_altitude <= 0.15 and horizontal_distance >= self.landing_error
        out_of_bounds = horizontal_distance > OUTER_RING_RADIUS_M + 5.0 or self.drone_position[2] > 4.0
        terminated = success or hard_landing or out_of_bounds
        truncated = self._step_count >= self.max_steps
        reward = 7.0 * (previous_distance - horizontal_distance) - 0.030 * horizontal_distance - 0.080 * float(np.square(action).sum())
        if horizontal_distance < 0.30:
            reward += 0.10
        reward += 0.35 * descent
        if success:
            reward += 100.0
        elif hard_landing:
            reward -= 50.0
        elif out_of_bounds:
            reward -= 20.0
        info = {
            "horizontal_error_m": horizontal_distance,
            "altitude_m": relative_altitude,
            "success": float(success),
            "hard_landing": float(hard_landing),
            "aligned_streak": float(self._aligned_streak),
            "target_speed_mps": float(np.linalg.norm(self._pad_velocity)),
            "trajectory_seed": float(self._motion.seed if self._motion else 0),
            "episode_steps": float(self._step_count),
            "physics_backend": "mujoco",
        }
        return self._observation(), float(reward), terminated, truncated, info

    def close(self) -> None:
        # MjModel/MjData are released by the Python bindings; explicit close
        # keeps the Gym interface symmetrical with rendering environments.
        return None
