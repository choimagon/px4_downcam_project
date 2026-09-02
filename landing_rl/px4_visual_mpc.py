"""Camera/PnP 3D velocity MPC used as a non-learning PX4 HIL baseline.

The controller receives only quantities available to the X500 companion:
camera/PnP target translation and target velocity, plus the PX4 EKF vehicle
velocity.  It returns a local-world three-axis velocity reference; PX4 still
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
    solve_period_s: float = 0.10
    # PX4's velocity loop and the HIL transport do not respond in one
    # 100-ms prediction interval.  Model the observed lag explicitly so
    # the optimizer leads a fast walking deck instead of crossing it.
    horizontal_velocity_response_time_constant_s: float = 0.38
    vertical_velocity_response_time_constant_s: float = 0.32
    position_weight: float = 28.0
    vertical_position_weight: float = 10.0
    relative_velocity_weight: float = 1.5
    vertical_relative_velocity_weight: float = 8.0
    command_change_weight: float = 0.04
    terminal_position_weight: float = 80.0
    terminal_vertical_position_weight: float = 24.0
    maximum_horizontal_speed_mps: float = 3.60
    maximum_climb_speed_mps: float = 0.75
    maximum_descent_speed_mps: float = 0.65
    vertical_standoff_tolerance_m: float = 0.001
    minimum_descent_toward_standoff_mps: float = 0.22
    horizontal_position_feedback_gain: float = 1.65
    final_horizontal_position_feedback_gain: float = 3.10
    final_horizontal_feedback_height_m: float = 0.50
    horizontal_relative_velocity_feedback_gain: float = 0.65
    horizontal_candidate_offset_limit_mps: float = 1.20
    # 75 move-blocking candidates (5 x 5 x 3) solve well within the 100-ms
    # companion period; a 9 x 9 x 5 Python loop starves PX4 HIL transport.
    horizontal_candidate_offset_count: int = 5
    vertical_candidate_offset_limit_mps: float = 0.30
    vertical_candidate_offset_count: int = 3


class CameraVelocityMpc:
    """Receding-horizon, move-blocking MPC for a moving visual marker.

    The predicted state is the world-frame QR-to-drone displacement and
    vehicle velocity.  For every candidate constant velocity target over
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
        if self.config.solve_period_s <= 0.0:
            raise ValueError("solve_period_s must be positive")
        if self.config.horizontal_velocity_response_time_constant_s <= 0.0:
            raise ValueError("horizontal_velocity_response_time_constant_s must be positive")
        if self.config.vertical_velocity_response_time_constant_s <= 0.0:
            raise ValueError("vertical_velocity_response_time_constant_s must be positive")
        if self.config.horizontal_candidate_offset_count < 3:
            raise ValueError("horizontal_candidate_offset_count must be at least three")
        if self.config.vertical_candidate_offset_count < 3:
            raise ValueError("vertical_candidate_offset_count must be at least three")
        self._previous_command = np.zeros(3, dtype=np.float64)
        self._last_solve_time_s = float("-inf")

    def reset(self) -> None:
        """Clear the rate-penalty reference after a visual-track loss."""
        self._previous_command[:] = 0.0
        self._last_solve_time_s = float("-inf")

    def command(
        self,
        *,
        target_minus_vehicle_xyz_m: np.ndarray,
        target_velocity_xyz_mps: np.ndarray,
        vehicle_velocity_xyz_mps: np.ndarray,
        timestamp_s: float | None = None,
    ) -> np.ndarray:
        """Return the first feasible three-axis velocity command.

        No simulator target truth is accepted here.  ``target_minus_vehicle``
        and ``target_velocity`` are camera/PnP estimates, and
        ``vehicle_velocity`` is the PX4 EKF state expressed in world axes.
        """
        error0 = np.asarray(target_minus_vehicle_xyz_m, dtype=np.float64).reshape(3)
        target_velocity = np.asarray(target_velocity_xyz_mps, dtype=np.float64).reshape(3)
        vehicle_velocity0 = np.asarray(vehicle_velocity_xyz_mps, dtype=np.float64).reshape(3)
        if not (
            np.all(np.isfinite(error0))
            and np.all(np.isfinite(target_velocity))
            and np.all(np.isfinite(vehicle_velocity0))
        ):
            self.reset()
            return np.zeros(3, dtype=np.float64)

        config = self.config
        if (
            timestamp_s is not None
            and math.isfinite(float(timestamp_s))
            and float(timestamp_s) - self._last_solve_time_s < config.solve_period_s - 1.0e-9
        ):
            # PX4 HIL transports sensors at 200 Hz while the camera/PnP
            # measurement and the predictive model are both 10 Hz.  Hold the
            # latest receding-horizon command between solver ticks instead of
            # spending a 5x5x3 candidate search on identical image data.
            return self._previous_command.copy()
        # A position-error-centred candidate lattice makes the finite search
        # expressive on a walking deck while its hard speed clip preserves the
        # PX4 companion safety boundary.
        # At the stock-skid approach height a fast Go2 can otherwise carry
        # the marker across the 5.5-cm physical touchdown tube before the
        # delayed PX4 velocity loop cancels the residual.  Increase only the
        # camera/PnP XY feedback there; the controller still chooses all
        # three axes by the same finite-horizon cost and safety limits.
        horizontal_feedback_gain = (
            config.final_horizontal_position_feedback_gain
            if -float(error0[2]) < config.final_horizontal_feedback_height_m
            else config.horizontal_position_feedback_gain
        )
        horizontal_relative_velocity = target_velocity[:2] - vehicle_velocity0[:2]
        nominal = target_velocity + np.array(
            (
                horizontal_feedback_gain * error0[0]
                + config.horizontal_relative_velocity_feedback_gain * horizontal_relative_velocity[0],
                horizontal_feedback_gain * error0[1]
                + config.horizontal_relative_velocity_feedback_gain * horizontal_relative_velocity[1],
                1.20 * error0[2],
            ),
            dtype=np.float64,
        )
        horizontal_offsets = np.linspace(
            -config.horizontal_candidate_offset_limit_mps,
            config.horizontal_candidate_offset_limit_mps,
            config.horizontal_candidate_offset_count,
            dtype=np.float64,
        )
        vertical_offsets = np.linspace(
            -config.vertical_candidate_offset_limit_mps,
            config.vertical_candidate_offset_limit_mps,
            config.vertical_candidate_offset_count,
            dtype=np.float64,
        )
        horizontal_response_alpha = 1.0 - math.exp(
            -config.prediction_dt_s / config.horizontal_velocity_response_time_constant_s
        )
        vertical_response_alpha = 1.0 - math.exp(
            -config.prediction_dt_s / config.vertical_velocity_response_time_constant_s
        )
        best_cost = float("inf")
        best_command = np.zeros(3, dtype=np.float64)
        for offset_x in horizontal_offsets:
            for offset_y in horizontal_offsets:
                for offset_z in vertical_offsets:
                    candidate = nominal + np.array((offset_x, offset_y, offset_z), dtype=np.float64)
                    horizontal_speed = float(np.linalg.norm(candidate[:2]))
                    if horizontal_speed > config.maximum_horizontal_speed_mps:
                        candidate[:2] *= config.maximum_horizontal_speed_mps / horizontal_speed
                    candidate[2] = float(
                        np.clip(
                            candidate[2],
                            -config.maximum_descent_speed_mps,
                            config.maximum_climb_speed_mps,
                        )
                    )
                    # ``error0[2]`` is expressed relative to the desired
                    # physical camera/skid standoff supplied by the caller.
                    # Above that standoff, a zero/climb candidate is not a
                    # feasible landing-progress solution.  Keep a small
                    # nonzero downwards element in the MPC feasible set; the
                    # outer visual safety governor still clips the final
                    # speed and may hold descent on invalid imagery.
                    if error0[2] < -config.vertical_standoff_tolerance_m:
                        candidate[2] = min(
                            candidate[2],
                            -config.minimum_descent_toward_standoff_mps,
                        )
                    predicted_error = error0.copy()
                    predicted_velocity = vehicle_velocity0.copy()
                    cost = config.command_change_weight * float(
                        np.dot(candidate - self._previous_command, candidate - self._previous_command)
                    )
                    for step in range(config.horizon_steps):
                        predicted_velocity[:2] += horizontal_response_alpha * (
                            candidate[:2] - predicted_velocity[:2]
                        )
                        predicted_velocity[2] += vertical_response_alpha * (
                            candidate[2] - predicted_velocity[2]
                        )
                        predicted_error += config.prediction_dt_s * (target_velocity - predicted_velocity)
                        position_weight_xy = (
                            config.terminal_position_weight
                            if step == config.horizon_steps - 1 else config.position_weight
                        )
                        position_weight_z = (
                            config.terminal_vertical_position_weight
                            if step == config.horizon_steps - 1 else config.vertical_position_weight
                        )
                        cost += position_weight_xy * float(np.dot(predicted_error[:2], predicted_error[:2]))
                        cost += position_weight_z * float(predicted_error[2] * predicted_error[2])
                        relative_velocity = target_velocity - predicted_velocity
                        cost += config.relative_velocity_weight * float(
                            np.dot(relative_velocity[:2], relative_velocity[:2])
                        )
                        cost += config.vertical_relative_velocity_weight * float(
                            relative_velocity[2] * relative_velocity[2]
                        )
                    if cost < best_cost:
                        best_cost = cost
                        best_command = candidate
        self._previous_command[:] = best_command
        if timestamp_s is not None and math.isfinite(float(timestamp_s)):
            self._last_solve_time_s = float(timestamp_s)
        return best_command.copy()
