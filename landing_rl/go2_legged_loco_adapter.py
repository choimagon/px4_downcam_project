"""Run the retrained legged-loco-compatible Go2 PPO inside the landing scene."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from stable_baselines3 import PPO

from .go2_legged_loco_environment import DEPLOYMENT_POLICY_ACTION_GAIN, STAND_POSE, _rpy

if TYPE_CHECKING:
    from .go2_qr_environment import Go2BackQrLandingEnv


class Go2LeggedLocoAdapter:
    """20 ms low-level policy bridge for a 5 ms Go2 landing simulation."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        deployment_action_gain: float = DEPLOYMENT_POLICY_ACTION_GAIN,
        action_limit: float = 1.0,
        observation_action_gain: float | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Missing Go2 legged-loco PPO: {self.model_path}")
        self.policy = PPO.load(self.model_path, device="cpu")
        if self.policy.observation_space.shape != (450,) or self.policy.action_space.shape != (12,):
            raise ValueError("Go2 legged-loco PPO must have 450 observations and 12 actions")
        if not 0.0 < deployment_action_gain <= 1.0:
            raise ValueError("deployment_action_gain must be in (0, 1]")
        if not 0.0 < action_limit <= 1.0:
            raise ValueError("action_limit must be in (0, 1]")
        self.deployment_action_gain = float(deployment_action_gain)
        self.action_limit = float(action_limit)
        self.observation_action_gain = float(
            deployment_action_gain if observation_action_gain is None else observation_action_gain
        )
        self.history: deque[np.ndarray] = deque(maxlen=9)
        self.delay: deque[np.ndarray] = deque(maxlen=5)
        self.last_action = np.zeros(12, dtype=np.float64)
        self.physics_tick = 0
        self.current_action = np.zeros(12, dtype=np.float64)

    def _single_observation(self, env: "Go2BackQrLandingEnv") -> np.ndarray:
        command = env._locomotion_command(float(env.data.time))
        return np.concatenate(
            (
                np.clip(env.data.cvel[env.base_id, :3], -8.0, 8.0),
                np.clip(_rpy(env.data.xmat[env.base_id]), -np.pi, np.pi),
                command,
                np.clip(env.data.qpos[env.go2_qposadr] - STAND_POSE, -2.0, 2.0),
                np.clip(env.data.qvel[env.go2_dofadr], -20.0, 20.0),
                self.last_action,
            )
        ).astype(np.float32)

    def reset(self, env: "Go2BackQrLandingEnv") -> None:
        self.history.clear()
        self.delay.clear()
        self.last_action[:] = 0.0
        self.current_action[:] = 0.0
        self.physics_tick = 0
        for _ in range(5):
            self.delay.append(np.zeros(12, dtype=np.float64))
        current = self._single_observation(env)
        for _ in range(9):
            self.history.append(current.copy())

    def _observation(self, env: "Go2BackQrLandingEnv") -> np.ndarray:
        current = self._single_observation(env)
        return np.concatenate((current, *self.history)).astype(np.float32)

    def control(self, env: "Go2BackQrLandingEnv") -> np.ndarray:
        """Return the learned residual target on every 5 ms physics tick."""
        if self.physics_tick % 4 == 0:
            action, _ = self.policy.predict(self._observation(env), deterministic=True)
            self.current_action = self.deployment_action_gain * np.clip(
                np.asarray(action, dtype=np.float64), -self.action_limit, self.action_limit
            )
            self.delay.append(self.current_action.copy())
            # The 450-D policy history contains the residual as it reaches
            # the joint target.  In the landing bridge terrain gain is
            # applied after this adapter, so record that same effective value
            # here rather than the unscaled request.
            self.last_action = self.observation_action_gain * self.current_action.copy()
            self.history.append(self._single_observation(env))
        self.physics_tick += 1
        return self.delay[0]
