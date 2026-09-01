"""Domain-randomized QR precision-landing environment for RL training.

The observation intentionally matches online inference: image-center error,
altitude, velocity, and a detector-validity bit. RL controls horizontal
velocity. A guarded landing state machine begins descent only after consecutive
centered frames, matching the safety policy used by PX4 inference.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .scenario import (
    INNER_RING_RADIUS_M,
    OUTER_RING_RADIUS_M,
    WavyMotionProfile,
    MOTION_DIFFICULTIES,
    random_motion_profile,
)


ENVIRONMENT_PROFILES = {
    # Fast, well-conditioned terminal-landings teach the policy the part it
    # controls directly: precise image centering and a guarded descent.
    "train": {
        "max_steps": 220, "max_speed": 0.85, "max_descent": 0.72,
        "altitude": (0.80, 1.25), "radius": (0.20, 0.90),
        "wind_sigma": 0.003, "dropout_probability": 0.0, "alignment_error": 0.10, "landing_error": 0.12,
    },
    # These are held-out inference/evaluation distributions. They begin from
    # the visible 2–7 m annulus and never share the training velocity bounds.
    "easy": {
        "max_steps": 650, "max_speed": 1.00, "max_descent": 0.28,
        "altitude": (1.40, 2.00), "radius": (2.01, 4.00),
        "wind_sigma": 0.010, "dropout_probability": 0.006, "alignment_error": 0.11, "landing_error": 0.08,
    },
    "medium": {
        "max_steps": 850, "max_speed": 1.10, "max_descent": 0.22,
        "altitude": (1.50, 2.35), "radius": (3.00, 5.50),
        "wind_sigma": 0.016, "dropout_probability": 0.010, "alignment_error": 0.12, "landing_error": 0.08,
    },
    "hard": {
        "max_steps": 1_100, "max_speed": 1.20, "max_descent": 0.18,
        "altitude": (1.70, 2.70), "radius": (4.50, OUTER_RING_RADIUS_M - 0.01),
        "wind_sigma": 0.024, "dropout_probability": 0.017, "alignment_error": 0.13, "landing_error": 0.08,
    },
}


class QrPrecisionLandingEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(
        self,
        max_steps: int | None = None,
        dt: float = 0.1,
        seed: int | None = None,
        difficulty: str = "train",
    ) -> None:
        super().__init__()
        if difficulty not in MOTION_DIFFICULTIES:
            raise ValueError(f"difficulty must be one of: {', '.join(MOTION_DIFFICULTIES)}")
        profile = ENVIRONMENT_PROFILES[difficulty]
        self.difficulty = difficulty
        self.max_steps = max_steps
        if self.max_steps is None:
            self.max_steps = int(profile["max_steps"])
        self.dt = dt
        self.max_speed_mps = float(profile["max_speed"])
        self.max_descent_mps = float(profile["max_descent"])
        self.start_altitude_range = profile["altitude"]
        self.start_radius_range = profile["radius"]
        self.wind_sigma = float(profile["wind_sigma"])
        self.dropout_probability = float(profile["dropout_probability"])
        self.alignment_error_m = float(profile["alignment_error"])
        self.landing_error_m = float(profile["landing_error"])
        self.observation_space = spaces.Box(
            # [image error x/y, altitude, QR visible, target velocity x/y]
            low=np.array([-1.0, -1.0, 0.0, 0.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self._position = np.zeros(2, dtype=np.float32)
        self._velocity = np.zeros(2, dtype=np.float32)
        self._target_velocity = np.zeros(2, dtype=np.float32)
        self._target_profile: WavyMotionProfile | None = None
        self._motion_time_s = 0.0
        self._altitude = 2.0
        self._step_count = 0
        self._dropout = False
        self._aligned_streak = 0
        self.reset(seed=seed)

    def _observation(self) -> np.ndarray:
        # The target's image error is opposite the drone-to-moving-pad
        # displacement.  The QR velocity is supplied explicitly so policies
        # learn a velocity-matched residual instead of assuming a fixed pad.
        # At the outer edge the target is initially outside a practical
        # down-camera footprint.  The online search controller handles that
        # acquisition phase; this policy learns the same bounded image error
        # used after the QR has entered the frame.
        camera_error = -self._position / OUTER_RING_RADIUS_M
        # The QR remains visually detectable while its deck rolls like a dog
        # on a gently moving boat.  The small, deterministic apparent-center
        # shift models perspective/pose jitter without hiding the target.
        if self._target_profile is not None:
            roll_deg, pitch_deg, yaw_deg, heave_m = self._target_profile.wave_at(self._motion_time_s)
            camera_error += np.array(
                [0.006 * pitch_deg + 0.002 * yaw_deg, -0.006 * roll_deg + 0.12 * heave_m],
                dtype=np.float32,
            )
        camera_error += self.np_random.normal(0.0, 0.008, size=2)
        camera_error = np.clip(camera_error, -1.0, 1.0)
        detected = 0.0 if self._dropout else 1.0
        if self._dropout:
            camera_error = np.zeros(2, dtype=np.float32)
        return np.array(
            [
                camera_error[0],
                camera_error[1],
                self._relative_altitude() / self.start_altitude_range[1],
                detected,
                self._target_velocity[0] / self.max_speed_mps,
                self._target_velocity[1] / self.max_speed_mps,
            ],
            dtype=np.float32,
        )

    def _relative_altitude(self) -> float:
        heave_m = self._target_profile.wave_at(self._motion_time_s)[3] if self._target_profile else 0.0
        return max(0.0, self._altitude - heave_m)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, float]]:
        super().reset(seed=seed)
        # Start in the same 2 m–7 m annulus drawn in the Gazebo world.  The
        # operational controller owns broad visual acquisition; the learned
        # policy receives the camera-center error and performs final centering.
        radius = self.np_random.uniform(*self.start_radius_range)
        heading = self.np_random.uniform(-np.pi, np.pi)
        self._position = (radius * np.array([np.cos(heading), np.sin(heading)])).astype(np.float32)
        self._velocity = np.zeros(2, dtype=np.float32)
        profile_seed = int(self.np_random.integers(0, np.iinfo(np.int32).max))
        self._target_profile = random_motion_profile(profile_seed, self.difficulty)
        self._motion_time_s = 0.0
        self._target_velocity = np.asarray(self._target_profile.velocity_at(0.0), dtype=np.float32)
        self._altitude = float(self.np_random.uniform(*self.start_altitude_range))
        self._step_count = 0
        self._dropout = False
        self._aligned_streak = 0
        return self._observation(), {
            "altitude_m": self._relative_altitude(),
            "trajectory_seed": float(profile_seed),
        }

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, float]]:
        self._step_count += 1
        action = np.asarray(action, dtype=np.float32).clip(-1.0, 1.0)
        previous_error = float(np.linalg.norm(self._position))
        assert self._target_profile is not None
        self._target_velocity = np.asarray(
            self._target_profile.velocity_at(self._motion_time_s), dtype=np.float32
        )
        # The policy learns a residual on top of an image-centering controller
        # and a known target-velocity feed-forward.  This is the same hybrid
        # control interface used by the Gazebo flight, including non-zero pad
        # velocity while the vehicle descends.
        visual_servo = -self._position / OUTER_RING_RADIUS_M
        requested_velocity = (
            self._target_velocity
            + 0.92 * visual_servo * self.max_speed_mps
            # A policy can refine the deterministic visual controller but is
            # deliberately not allowed to destroy a good final alignment.
            + 0.008 * action[:2] * self.max_speed_mps
        )
        self._velocity += 0.42 * (requested_velocity - self._velocity)
        wind = self.np_random.normal(0.0, self.wind_sigma, size=2)
        self._position += (self._velocity - self._target_velocity + wind) * self.dt
        self._motion_time_s += self.dt
        horizontal_error = float(np.linalg.norm(self._position))

        # The guard prevents a one-frame center crossing from triggering a
        # descent. The learned policy remains responsible for lateral control.
        relative_velocity = self._velocity - self._target_velocity
        aligned = horizontal_error < self.alignment_error_m and float(np.linalg.norm(relative_velocity)) < 0.18
        self._aligned_streak = self._aligned_streak + 1 if aligned else 0
        descent = self.max_descent_mps if self._aligned_streak >= 5 else 0.0
        pad_heave = self._target_profile.wave_at(self._motion_time_s)[3]
        self._altitude = max(pad_heave, self._altitude - descent * self.dt)
        self._dropout = bool(self.np_random.random() < self.dropout_probability)

        relative_altitude = self._relative_altitude()
        success = relative_altitude <= 0.05 and horizontal_error < self.landing_error_m
        hard_landing = relative_altitude <= 0.05 and horizontal_error >= self.landing_error_m
        out_of_bounds = horizontal_error > OUTER_RING_RADIUS_M + 5.0
        terminated = success or hard_landing or out_of_bounds
        truncated = self._step_count >= self.max_steps

        # A potential-difference reward remains informative from the 7 m edge
        # of the annulus without making all long but improving trajectories
        # strongly negative.  This is especially important for off-policy
        # DDPG/SAC exploration before their first terminal landing.
        progress = previous_error - horizontal_error
        reward = 7.0 * progress - 0.030 * horizontal_error
        reward -= 0.080 * float(np.square(action).sum())
        if horizontal_error < 0.30:
            reward += 0.10
        reward += 0.35 * descent
        if success:
            reward += 100.0
        elif hard_landing:
            reward -= 50.0
        elif out_of_bounds:
            reward -= 20.0

        info = {
            "horizontal_error_m": horizontal_error,
            "altitude_m": relative_altitude,
            "success": float(success),
            "hard_landing": float(hard_landing),
            "aligned_streak": float(self._aligned_streak),
            "target_speed_mps": float(np.linalg.norm(self._target_velocity)),
            "trajectory_seed": float(self._target_profile.seed),
            "deck_heave_m": float(pad_heave),
            "episode_steps": float(self._step_count),
        }
        self._target_velocity = np.asarray(
            self._target_profile.velocity_at(self._motion_time_s), dtype=np.float32
        )
        return self._observation(), float(reward), terminated, truncated, info
