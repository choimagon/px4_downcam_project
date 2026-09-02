"""Camera/PnP velocity MPC used as a non-learning PX4 HIL baseline.

The controller receives only quantities available to the X500 companion:
camera/PnP target translation and target velocity, plus the PX4 EKF vehicle
velocity.  It returns a local-world horizontal velocity reference; PX4 still
owns attitude, collective thrust, motor allocation and all motor outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class VisualMpcConfig:
    """Finite-horizon model and cost weights for the camera-track MPC."""

    horizon_steps: int = 8
    prediction_dt_s: float = 0.10
    # PX4's velocity loop and the HIL transport do not respond in one
    # 100-ms prediction interval.  Model the observed lag explicitly so
    # the optimizer leads a fast walking deck instead of crossing it.
    velocity_response_time_constant_s: float = 0.38
    position_weight: float = 8.0
    relative_velocity_weight: float = 7.0
    command_change_weight: float = 0.04
    terminal_position_weight: float = 20.0
    maximum_speed_mps: float = 3.60
    candidate_offset_limit_mps: float = 1.20
    candidate_offset_count: int = 9


class CameraVelocityMpc:
    """Receding-horizon, move-blocking MPC for a moving visual marker.

    The predicted state is the world-frame QR-to-drone horizontal displacement
    and vehicle velocity.  For every candidate constant velocity target over
    the horizon, a first-order PX4 velocity-loop response is simulated.  The
    lowest quadratic tracking cost wins; only the first reference is emitted,
    then the problem is solved again at the next control update.
    """

    def __init__(self, config: VisualMpcConfig | None = None) -> None:
        self.config = config or VisualMpcConfig()
        if self.config.horizon_steps <= 0:
            raise ValueError("horizon_steps must be positive")
        if self.config.prediction_dt_s <= 0.0:
            raise ValueError("prediction_dt_s must be positive")
        if self.config.velocity_response_time_constant_s <= 0.0:
            raise ValueError("velocity_response_time_constant_s must be positive")
        if self.config.candidate_offset_count < 3:
            raise ValueError("candidate_offset_count must be at least three")
        self._previous_command = np.zeros(2, dtype=np.float64)

    def reset(self) -> None:
        """Clear the rate-penalty reference after a visual-track loss."""
        self._previous_command[:] = 0.0

    def command(
        self,
        *,
        target_minus_vehicle_xy_m: np.ndarray,
        target_velocity_xy_mps: np.ndarray,
        vehicle_velocity_xy_mps: np.ndarray,
    ) -> np.ndarray:
        """Return the first feasible horizontal velocity command.

        No simulator target truth is accepted here.  ``target_minus_vehicle``
        and ``target_velocity`` are camera/PnP estimates, and
        ``vehicle_velocity`` is the PX4 EKF state expressed in world axes.
        """
        error0 = np.asarray(target_minus_vehicle_xy_m, dtype=np.float64).reshape(2)
        target_velocity = np.asarray(target_velocity_xy_mps, dtype=np.float64).reshape(2)
        vehicle_velocity0 = np.asarray(vehicle_velocity_xy_mps, dtype=np.float64).reshape(2)
        if not (
            np.all(np.isfinite(error0))
            and np.all(np.isfinite(target_velocity))
            and np.all(np.isfinite(vehicle_velocity0))
        ):
            self.reset()
            return np.zeros(2, dtype=np.float64)

        config = self.config
        # A position-error-centred candidate lattice makes the finite search
        # expressive on a walking deck while its hard speed clip preserves the
        # PX4 companion safety boundary.
        nominal = target_velocity + 1.65 * error0
        offsets = np.linspace(
            -config.candidate_offset_limit_mps,
            config.candidate_offset_limit_mps,
            config.candidate_offset_count,
            dtype=np.float64,
        )
        response_alpha = 1.0 - math.exp(
            -config.prediction_dt_s / config.velocity_response_time_constant_s
        )
        best_cost = float("inf")
        best_command = np.zeros(2, dtype=np.float64)
        for offset_x in offsets:
            for offset_y in offsets:
                candidate = nominal + np.array((offset_x, offset_y), dtype=np.float64)
                speed = float(np.linalg.norm(candidate))
                if speed > config.maximum_speed_mps:
                    candidate *= config.maximum_speed_mps / speed
                predicted_error = error0.copy()
                predicted_velocity = vehicle_velocity0.copy()
                cost = config.command_change_weight * float(
                    np.dot(candidate - self._previous_command, candidate - self._previous_command)
                )
                for step in range(config.horizon_steps):
                    predicted_velocity += response_alpha * (candidate - predicted_velocity)
                    predicted_error += config.prediction_dt_s * (target_velocity - predicted_velocity)
                    position_weight = (
                        config.terminal_position_weight
                        if step == config.horizon_steps - 1 else config.position_weight
                    )
                    cost += position_weight * float(np.dot(predicted_error, predicted_error))
                    relative_velocity = target_velocity - predicted_velocity
                    cost += config.relative_velocity_weight * float(
                        np.dot(relative_velocity, relative_velocity)
                    )
                if cost < best_cost:
                    best_cost = cost
                    best_command = candidate
        self._previous_command[:] = best_command
        return best_command.copy()
