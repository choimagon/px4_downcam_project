#!/usr/bin/env python3
"""Record a flat MuJoCo QR landing driven by a separate real PX4 SITL.

The X500's MuJoCo IMU, barometer and GPS are sent to PX4's
``simulator_mavlink`` HIL interface.  PX4 EKF2 supplies the state used by the
camera-only companion mission, and PX4's HIL actuator outputs are the *only*
motor commands converted to an X500 body wrench.  PPO, DDPG, or SAC can be
evaluated as the same bounded QR-camera residual policy used by the existing
flat comparison.

This runner intentionally supports flat terrain only.  It does not alter a
PX4 checkout or a user's Gazebo process: ``Px4MujocoHilSession`` creates an
isolated temporary PX4 rootfs for each run and preserves its ULog beside the
recording for audit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from types import MethodType

os.environ.setdefault("MUJOCO_GL", "egl")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np
import onnxruntime as ort

from landing_rl.go2_onnx_inference import (
    animate_propellers,
    compose_dual_view,
    follow_camera,
)
from landing_rl.go2_qr_environment import (
    FINAL_PRECISION_TARGET_HEIGHT_M,
    Go2BackQrLandingEnv,
    TRACKING_MEMORY_S,
)
from landing_rl.px4_mujoco_hil import (
    Px4MujocoHilSession,
    px4_local_ned_to_world_enu,
    rpy_ned_to_world_from_body_flu,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("ppo", "ddpg", "sac"), default="ppo")
    parser.add_argument(
        "--onnx-model",
        type=Path,
        default=None,
        help="Policy ONNX; defaults to the canonical artifact for --algorithm.",
    )
    parser.add_argument("--difficulty", choices=("easy", "medium", "hard"), default="easy")
    parser.add_argument(
        "--locomotion-model",
        type=Path,
        default=PROJECT_ROOT / "models" / "go2_legged_loco_ppo.zip",
        help="Required low-level Go2 PPO used for physical, foot-contact locomotion.",
    )
    parser.add_argument(
        "--go2-policy-action-gain",
        type=float,
        default=0.50,
        help="Learned Go2 PPO residual gain around the physical trot prior; does not scale route speed.",
    )
    parser.add_argument(
        "--go2-speed-scale",
        type=float,
        default=1.0,
        help="Scale the commanded physical Go2 walking speed; 1.0 preserves the trained difficulty profile.",
    )
    parser.add_argument(
        "--go2-turn-scale",
        type=float,
        default=1.0,
        help="Scale route-curvature amplitude; 1.0 preserves the trained difficulty profile.",
    )
    parser.add_argument(
        "--go2-motion-start-delay-s",
        type=float,
        default=0.0,
        help="Optional visual-acquisition lead-in before Go2 begins its continuous walking route.",
    )
    parser.add_argument(
        "--capture-radius-m",
        type=float,
        default=0.35,
        help="Camera/PnP horizontal capture radius that permits guarded descent.",
    )
    parser.add_argument(
        "--search-altitude-world-m",
        type=float,
        default=1.80,
        help="PX4 camera-search altitude in MuJoCo world coordinates.",
    )
    parser.add_argument(
        "--flight-policy-residual-gain",
        type=float,
        default=0.020,
        help="Maximum horizontal contribution (m/s per normalized unit) of the ONNX policy residual.",
    )
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--video-file", type=Path, required=True)
    parser.add_argument("--snapshot-file", type=Path, required=True)
    parser.add_argument("--metrics-file", type=Path, required=True)
    parser.add_argument("--trace-file", type=Path, required=True)
    parser.add_argument("--px4-log-file", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=650)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _draw_lines(frame: np.ndarray, lines: tuple[str, ...], *, origin: tuple[int, int], scale: float) -> np.ndarray:
    image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    x, y = origin
    for index, line in enumerate(lines):
        position = (x, y + 22 * index)
        cv2.putText(image, line, position, cv2.FONT_HERSHEY_SIMPLEX, scale, (3, 8, 14), 3, cv2.LINE_AA)
        cv2.putText(image, line, position, cv2.FONT_HERSHEY_SIMPLEX, scale, (235, 247, 255), 1, cv2.LINE_AA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def render_dual_view(
    environment: Go2BackQrLandingEnv,
    *,
    renderer: mujoco.Renderer,
    down_camera_option: mujoco.MjvOption,
    propeller_ids: np.ndarray,
    propeller_base_quaternions: np.ndarray,
    mission: "Px4VisionMission",
    algorithm: str,
) -> np.ndarray:
    """Capture synchronized third-person and attached downward camera panels."""
    animate_propellers(environment, propeller_ids, propeller_base_quaternions)
    error = float(np.linalg.norm(environment._horizontal_error()))
    altitude = float(environment._relative_altitude())
    motor_mean = float(np.mean(mission.last_motor_outputs))
    go2_speed = float(np.linalg.norm(environment.data.qvel[:2]))
    go2_command_speed = float(np.linalg.norm(environment._path_command(float(environment.data.time))[:2]))
    renderer.update_scene(environment.data, camera=follow_camera(environment))
    third = renderer.render().copy()
    third = _draw_lines(
        third,
        (
            f"PX4 SITL EKF2 + MAVLink HIL | MuJoCo flat | {algorithm.upper()} QR residual",
            f"t={environment.data.time:05.1f}s  QR error={error:.3f}m  rel altitude={altitude:.2f}m",
            f"PX4: {'ARMED' if mission.session.armed else 'DISARMED'} | OFFBOARD={'ON' if mission.session.offboard_active else 'OFF'} | motor={motor_mean:.2f}",
            f"EKF2 innovation: XY={mission.horizontal_innovation_ratio:.4f}  Z={mission.vertical_innovation_ratio:.4f}",
            f"VISION: {'QR TRACKING' if environment._qr_detected else 'MISSION SEARCH'} | descent={'ON' if mission.landing_committed else 'HOLD'}",
            f"GO2: WALKING | actual={go2_speed:.2f} m/s  command={go2_command_speed:.2f} m/s  path={environment._path_length:.2f} m",
            "Motor physics: PX4 HIL_ACTUATOR_CONTROLS only (no direct flight wrench)",
        ),
        origin=(18, 28),
        scale=0.46,
    )
    renderer.update_scene(environment.data, camera="down_camera", scene_option=down_camera_option)
    down_wide = renderer.render().copy()
    down = down_wide[:, 320:960].copy()
    down = _draw_lines(
        down,
        (
            f"ATTACHED DOWN CAMERA | {'QR DETECTED' if environment._qr_detected else 'SEARCHING'}",
            f"QR error={error:.3f}m | depth={environment._qr_depth if environment._qr_detected else 0.0:.2f}m",
            f"PX4 EKF2 vertical speed={environment._onboard_velocity()[2]:+.2f}m/s",
            f"{algorithm.upper()} input: QR centre/depth/rate + PX4 vertical velocity",
        ),
        origin=(18, 30),
        scale=0.41,
    )
    height, width = down.shape[:2]
    marker_colour = (55, 235, 255)
    image = cv2.cvtColor(down, cv2.COLOR_RGB2BGR)
    cv2.drawMarker(image, (width // 2, height // 2), marker_colour, cv2.MARKER_CROSS, 28, 2, cv2.LINE_AA)
    return compose_dual_view(third, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


class Px4VisionMission:
    """Camera/RL companion command producer with PX4-only motor authority."""

    def __init__(
        self,
        environment: Go2BackQrLandingEnv,
        session: Px4MujocoHilSession,
        *,
        start_world_enu_m: np.ndarray,
        hil_time_base_us: int,
        flight_policy_residual_gain: float,
    ) -> None:
        self.environment = environment
        self.session = session
        self.start_world_enu_m = np.asarray(start_world_enu_m, dtype=np.float64).copy()
        self.hil_time_base_us = int(hil_time_base_us)
        self.flight_policy_residual_gain = float(flight_policy_residual_gain)
        self.last_motor_outputs = np.zeros(4, dtype=np.float64)
        self.landing_committed = False
        self._alignment_streak = 0
        self._precision_streak = 0
        self.last_desired_world_velocity = np.zeros(3, dtype=np.float64)
        # PX4 is a separate real-time process while MuJoCo advances in fixed
        # 5-ms physics increments.  Pace against an absolute deadline, not a
        # fresh ``sleep(5 ms)`` each tick, so ONNX inference cost cannot make
        # PPO/DDPG/SAC receive different effective PX4 sample periods.
        self._next_hil_wall_deadline = time.monotonic()

    @property
    def horizontal_innovation_ratio(self) -> float:
        value = self.session.ekf.horizontal_innovation_ratio
        return float(value) if value is not None and np.isfinite(value) else float("nan")

    @property
    def vertical_innovation_ratio(self) -> float:
        value = self.session.ekf.vertical_innovation_ratio
        return float(value) if value is not None and np.isfinite(value) else float("nan")

    def update_ekf_estimate(self, *, force: bool = False) -> None:
        """Publish real PX4 EKF2 state into the environment's onboard cache."""
        del force
        if not self.session.ekf_ready:
            return
        assert self.session.ekf.local_position_ned_m is not None
        assert self.session.ekf.local_velocity_ned_mps is not None
        assert self.session.ekf.attitude_rpy_ned_rad is not None
        self.environment._estimated_position[:] = px4_local_ned_to_world_enu(
            start_world_enu_m=self.start_world_enu_m,
            local_ned_m=self.session.ekf.local_position_ned_m,
        )
        self.environment._estimated_velocity[:] = np.array(
            (
                self.session.ekf.local_velocity_ned_mps[0],
                -self.session.ekf.local_velocity_ned_mps[1],
                -self.session.ekf.local_velocity_ned_mps[2],
            ),
            dtype=np.float64,
        )
        self.environment._estimated_rotation[:] = rpy_ned_to_world_from_body_flu(
            self.session.ekf.attitude_rpy_ned_rad
        )
        # This is a rotated onboard gyro measurement, not target/world state.
        gyro_body = self.environment._sensor("drone_gyro")
        self.environment._estimated_angular_velocity[:] = self.environment._estimated_rotation @ gyro_body

    def _desired_world_velocity(self, action: np.ndarray, *, update_alignment: bool) -> np.ndarray:
        env = self.environment
        own_position = env._onboard_position()
        own_velocity = env._onboard_velocity()
        rotation = env._onboard_rotation()
        action = np.asarray(action, dtype=np.float64).clip(-1.0, 1.0)
        detected = bool(env._qr_detected)
        # The camera is sampled at 30 Hz and has the configured visual
        # dropout probability.  A single missing frame must not cause PX4 to
        # discard the last *camera-derived* target estimate and fly back into
        # a blind-search pattern.  This uses exactly the same short visual
        # memory as the MuJoCo sensor-contract environment; it never reads
        # Go2 position, velocity, route, or any simulator-only target state.
        time_since_seen = float(env.data.time) - float(env._last_qr_seen_time)
        visual_memory_valid = time_since_seen <= TRACKING_MEMORY_S
        tracking = detected or visual_memory_valid

        if tracking:
            if detected:
                relative_world = rotation @ env._qr_translation_body
            else:
                predicted_target = (
                    env._qr_target_position_world
                    + max(0.0, time_since_seen) * env._qr_target_velocity_world
                )
                relative_world = predicted_target - own_position
            horizontal_error = float(np.linalg.norm(relative_world[:2]))
            relative_height = max(0.0, -float(relative_world[2]))
            target_velocity = env._qr_target_velocity_world.copy()
            policy_world = rotation[:2, :2] @ action
            # The actor is a trim policy, not a licence to command a blind
            # sideways departure from an observed QR.  Project its requested
            # horizontal residual onto the camera/PnP inward ray: PPO, DDPG,
            # and SAC can accelerate visual centring, while an outward or
            # tangential suggestion is rejected by the same safety filter.
            if horizontal_error > 1.0e-6:
                inward_direction = relative_world[:2] / horizontal_error
                inward_component = max(0.0, float(policy_world @ inward_direction))
                policy_world = inward_component * inward_direction
            else:
                policy_world = np.zeros(2, dtype=np.float64)
            # The RL policy remains a strictly bounded input residual.  The visual
            # servo owns the coarse pursuit and the PX4 position controller
            # owns all attitude/collective/motor output.
            # Follow the measured marker velocity, then turn up camera-PnP
            # position feedback in the last half-metre.  The former 1.45x
            # feed-forward passed the walking deck at low height; a measured
            # 1.0x feed-forward with a stronger centring term keeps the rail
            # contact point over the physical QR board.  Every term here is
            # camera/PnP or the bounded policy residual, never Go2 state.
            lateral_gain = 3.10 if relative_height < 0.50 else 1.35
            velocity_lead = 1.12 if relative_height < 0.60 else 1.0
            # Camera/PnP supplies both the target position and its filtered
            # relative velocity.  Add a bounded D term at stock-skid height
            # so the PX4 velocity loop matches the moving QR deck instead of
            # crossing its centre at residual speed.  It is entirely a drone
            # perception/EKF control term; Go2 state and Go2 control are not
            # read or modified here.
            relative_velocity_xy = np.clip(env._qr_relative_velocity_world[:2], -0.90, 0.90)
            velocity_damping = 0.65 if relative_height < 0.50 else 0.0
            desired_xy = (
                velocity_lead * target_velocity[:2]
                + lateral_gain * relative_world[:2]
                + velocity_damping * relative_velocity_xy
                + self.flight_policy_residual_gain * policy_world
            )
            relative_speed = float(np.linalg.norm(env._qr_relative_velocity_world[:2]))
            # PnP frame-to-frame velocity is deliberately noisy at the
            # 30-Hz camera boundary.  Using it as a hard descent gate kept a
            # correctly velocity-matched walking deck at 2.6 m altitude
            # forever: image error stayed near 0.13 m but the estimated
            # relative-speed threshold flickered.  Start a *slow, still
            # camera-centred* descent once the marker remains inside a
            # conservative stage-configured high-altitude capture tube.  At
            # the search height this is deliberately a coarse acquisition gate,
            # not a touchdown tolerance: the lateral controller keeps
            # centring during descent and the physical touchdown gate
            # below remains stricter (two skids, <55 mm QR error and <0.4
            # m/s deck-relative speed), so this is not a contact shortcut.
            aligned = (
                detected
                and horizontal_error < float(env.capture_radius_m)
                and relative_speed < 1.20
            )
            if update_alignment:
                self._alignment_streak = self._alignment_streak + 1 if aligned else 0
                if self._alignment_streak >= 5:
                    self.landing_committed = True
            # A 35-cm acquisition gate is intentionally not a landing gate.
            # At stock-skid height, retract to a small climb/hold if the QR
            # leaves the 3.5-cm PnP tube; then re-centre before resuming the
            # last centimetres of descent.  This mirrors the base environment
            # and prevents a side skid from crossing the visible QR deck.
            # Two consecutive camera frames inside the 6-cm approach tube
            # establish a measured precision lock.  The former four-frame,
            # 3.5-cm condition made the PX4 aircraft loiter at the stock-skid
            # height while a moving Go2 deck passed underneath it.  This is
            # still only an approach permission: physical touchdown remains
            # independently scored with both skids, <=55 mm QR error and the
            # measured relative-speed/contact thresholds.
            if update_alignment:
                if (
                    detected
                    and horizontal_error <= 0.060
                    and relative_speed <= 0.65
                ):
                    self._precision_streak += 1
                else:
                    self._precision_streak = 0
            precision_ready = self._precision_streak >= 2
            # After the measured lock has started final descent, retain it
            # inside the physical stock-skid capture envelope.  The success
            # scorer requires <55 mm at <=245 mm; forcing a new <35 mm gate
            # at 24 cm was needlessly climbing away from an already-safe
            # moving-deck contact.
            touchdown_capture = (
                relative_height <= 0.28
                and horizontal_error < 0.055
                and relative_speed <= 0.45
            )
            final_descent_allowed = precision_ready or touchdown_capture
            if self.landing_committed and detected:
                # Fast camera-tracked approach continues down to 32 cm.  The
                # stock skids still have safe clearance there, so demanding a
                # sub-5.5-cm touchdown tube at 38 cm only made the aircraft hover
                # beside a fast deck.  Below 30 cm it holds/recentres, then
                # performs the deliberately guarded final contact.
                if relative_height <= 0.32 and not final_descent_allowed:
                    # At the measured skid plane do not blindly climb away
                    # from a possible first-rail contact on a 1--5 cm PnP
                    # fluctuation.  Hold altitude and re-centre from camera
                    # only; climb is reserved for a genuine >10 cm miss.
                    recovery_climb = (
                        0.18
                        if relative_height < 0.25 and horizontal_error > 0.12
                        else 0.0
                    )
                    # A flat QR deck has no commanded vertical trajectory.
                    # PnP depth-difference velocity can contain camera
                    # jitter while the Go2 trots, so never feed that noisy
                    # estimate into the PX4 vertical setpoint.  Hold (or
                    # recover) from the measured relative height only.
                    desired_z = recovery_climb
                else:
                    if relative_height > 0.85:
                        descent = 0.50
                    elif relative_height > 0.38:
                        descent = 0.30
                    elif relative_height > FINAL_PRECISION_TARGET_HEIGHT_M + 0.004:
                        # Close the final centimetres before the moving deck
                        # exits the camera-centred lock.  The 0.22 m/s command
                        # is a PX4 velocity setpoint (not a direct force) and
                        # the physical success gate still rejects a one-skid
                        # or off-centre contact.
                        descent = 0.22
                    else:
                        descent = max(0.0, 3.5 * (relative_height - FINAL_PRECISION_TARGET_HEIGHT_M))
                    desired_z = -descent
            else:
                # Keep search/track altitude until the camera-based landing
                # gate is committed; the moving deck's PnP depth rate is not
                # a flight-height command.
                desired_z = 0.0
        else:
            desired_xy = env._search_velocity(own_position)
            desired_z = float(np.clip(1.0 * (env._search_altitude - own_position[2]), -0.30, 0.30))
            self._alignment_streak = 0
            self._precision_streak = 0
        horizontal_speed = float(np.linalg.norm(desired_xy))
        # PX4 can track a 3 m/s camera-search command after it is in steady
        # flight, but an airborne-start HIL craft must first establish a
        # stable attitude.  Limit the blind-search leg to 1.2 m/s and QR
        # pursuit to 3.6 m/s; this is a mission command limit, not a motor
        # or force shortcut.
        horizontal_limit = 3.6 if tracking else 1.2
        if horizontal_speed > horizontal_limit:
            desired_xy *= horizontal_limit / horizontal_speed
        desired = np.array((desired_xy[0], desired_xy[1], desired_z), dtype=np.float64)
        # Offboard body-frame velocity control has its own PX4 acceleration
        # limits.  This final cap prevents sending unachievable camera-search
        # commands when the target first enters the FOV.
        desired[2] = float(np.clip(desired[2], -0.65, 0.75))
        self.last_desired_world_velocity[:] = desired
        return desired

    def apply_px4_control(self, action: np.ndarray, *, update_alignment: bool) -> None:
        """Send sensor data/setpoint, then apply only PX4's motor wrench."""
        env = self.environment
        self.update_ekf_estimate()
        desired_world_velocity = self._desired_world_velocity(action, update_alignment=update_alignment)
        physical_rotation = env.data.xmat[env.drone_id].reshape(3, 3).copy()
        hil_time_us = self.hil_time_base_us + int(round(float(env.data.time) * 1.0e6))
        acceleration_body = env._sensor("drone_accelerometer")
        if float(env.data.time) <= float(env.model.opt.timestep) and np.linalg.norm(acceleration_body) < 1.0e-6:
            acceleration_body = np.array(
                (0.0, 0.0, abs(float(env.model.opt.gravity[2]))), dtype=np.float64
            )
        self.session.send_hil_sample(
            time_usec=hil_time_us,
            position_enu_m=env._sensor("drone_gps_position"),
            velocity_enu_mps=env._sensor("drone_gps_velocity"),
            acceleration_body_flu_mps2=acceleration_body,
            angular_velocity_body_flu_radps=env._sensor("drone_gyro"),
        )
        # MuJoCo's local world is north/east/up, exactly the HIL local-frame
        # counterpart of PX4's north/east/down.  Keep this companion command
        # in NED so the PX4 receiver does not depend on a delayed body-yaw
        # transform; no force, torque or direct actuator is sent here.
        self.session.send_local_velocity_ned(
            float(desired_world_velocity[0]),
            float(-desired_world_velocity[1]),
            float(-desired_world_velocity[2]),
        )
        # The HIL timebase advances at the MuJoCo 5 ms native step.  Keep
        # wall pacing equally real so PX4's separate process can consume the
        # sensor sample before this exact physical interval is integrated.
        # An absolute deadline removes model-dependent ONNX timing drift.
        self._next_hil_wall_deadline += float(env.model.opt.timestep)
        sleep_s = self._next_hil_wall_deadline - time.monotonic()
        if sleep_s > 0.0:
            time.sleep(sleep_s)
        else:
            self._next_hil_wall_deadline = time.monotonic()
        self.session.pump()
        force_world, torque_world, motor_thrusts = self.session.motor_wrench_world(
            world_from_body_flu=physical_rotation,
            vehicle_mass_kg=env.drone_mass,
        )
        env.data.xfrc_applied[env.drone_id] = 0.0
        env.data.xfrc_applied[env.drone_id, :3] = force_world
        env.data.xfrc_applied[env.drone_id, 3:6] = torque_world
        self.last_motor_outputs[:] = self.session.diagnostics.last_actuator_controls


def warm_and_arm_px4(
    session: Px4MujocoHilSession,
    environment: Go2BackQrLandingEnv,
    *,
    timeout_s: float = 18.0,
) -> int:
    """Warm real sensors, verify EKF2 innovations, then enable Offboard/arm."""
    start = time.monotonic()
    mode_sent = False
    last_arm_request_s = float("-inf")
    last_hil_time_us = 0
    while time.monotonic() - start < timeout_s:
        elapsed = time.monotonic() - start
        last_hil_time_us = int(elapsed * 1.0e6)
        session.send_hil_sample(
            time_usec=last_hil_time_us,
            position_enu_m=environment._sensor("drone_gps_position"),
            velocity_enu_mps=environment._sensor("drone_gps_velocity"),
            # ``mj_forward`` correctly leaves a free-body accelerometer at
            # zero before the first dynamics step.  PX4 pre-arm is a static,
            # supported-aircraft calibration phase, so supply its physically
            # correct +g specific force until the first PX4 motor wrench is
            # applied and the native MuJoCo IMU starts producing it.
            acceleration_body_flu_mps2=np.array(
                (0.0, 0.0, abs(float(environment.model.opt.gravity[2]))), dtype=np.float64
            ),
            angular_velocity_body_flu_radps=environment._sensor("drone_gyro"),
        )
        session.send_gcs_heartbeat()
        # A gentle upward setpoint lets PX4 pass its normal airborne/takeoff
        # state transition while the MuJoCo world is still paused for HIL
        # initialization.  The first live physics tick immediately replaces
        # it with the camera mission setpoint.
        session.send_body_velocity(0.0, 0.0, -0.25)
        xy = session.ekf.horizontal_innovation_ratio
        z = session.ekf.vertical_innovation_ratio
        stable = (
            xy is not None
            and z is not None
            and np.isfinite(xy)
            and np.isfinite(z)
            and max(float(xy), float(z)) < 0.5
        )
        if stable and not mode_sent:
            mode_sent = session.request_offboard_mode()
        # Commander may publish a final "Ready for takeoff" health update
        # immediately after Offboard acceptance.  Re-request at 1 Hz until
        # it explicitly arms; this is ordinary MAVLink retry behaviour, not
        # a bypass of the health checks.
        if stable and session.offboard_active and not session.armed and elapsed - last_arm_request_s >= 1.0:
            session.request_arm()
            last_arm_request_s = elapsed
        # The world is intentionally paused while PX4 completes its native
        # arming/spool transition.  The X500 starts airborne, so releasing
        # MuJoCo while PX4 is still emitting only the 0.1% idle command would
        # manufacture a free-fall unrelated to the policy or EKF.  Resume
        # only once the genuine PX4 motor output has crossed hover spool-up.
        if session.armed and float(np.mean(session.diagnostics.last_actuator_controls)) > 0.20:
            return last_hil_time_us + int(round(environment.model.opt.timestep * 1.0e6))
        time.sleep(float(environment.model.opt.timestep))
    raise RuntimeError(
        "PX4 HIL warm-up did not arm: "
        f"acks={session.diagnostics.command_ack_results}, "
        f"EKF xy={session.ekf.horizontal_innovation_ratio}, z={session.ekf.vertical_innovation_ratio}, "
        f"status={session.diagnostics.last_status_text!r}"
    )


def main() -> None:
    args = parse_args()
    if args.onnx_model is None:
        args.onnx_model = Path(f"artifacts/rl_training/onnx_go2/{args.algorithm}_go2_back_qr.onnx")
    if not args.onnx_model.is_file():
        raise SystemExit(f"Missing {args.algorithm.upper()} ONNX model: {args.onnx_model}")
    if args.max_steps <= 0:
        raise SystemExit("--max-steps must be positive")
    if not args.locomotion_model.is_file():
        raise SystemExit(f"Missing Go2 low-level locomotion model: {args.locomotion_model}")
    if not 0.10 <= args.go2_speed_scale <= 1.0:
        raise SystemExit("--go2-speed-scale must be within [0.10, 1.0]")
    if not 0.10 <= args.go2_turn_scale <= 1.0:
        raise SystemExit("--go2-turn-scale must be within [0.10, 1.0]")
    if not 0.0 <= args.go2_motion_start_delay_s <= 5.0:
        raise SystemExit("--go2-motion-start-delay-s must be within [0.0, 5.0]")
    if not 0.05 <= args.capture_radius_m <= 0.45:
        raise SystemExit("--capture-radius-m must be within [0.05, 0.45]")
    if not 1.30 <= args.search_altitude_world_m <= 2.72:
        raise SystemExit("--search-altitude-world-m must be within [1.30, 2.72]")
    if not 0.10 <= args.go2_policy_action_gain <= 0.50:
        raise SystemExit("--go2-policy-action-gain must be within [0.10, 0.50]")
    if not 0.0 <= args.flight_policy_residual_gain <= 0.08:
        raise SystemExit("--flight-policy-residual-gain must be within [0.0, 0.08]")
    for path in (args.video_file, args.snapshot_file, args.metrics_file, args.trace_file, args.px4_log_file):
        path.parent.mkdir(parents=True, exist_ok=True)

    policy = ort.InferenceSession(str(args.onnx_model), providers=["CPUExecutionProvider"])
    policy_input = policy.get_inputs()[0].name
    policy_output = policy.get_outputs()[0].name
    environment = Go2BackQrLandingEnv(
        difficulty=args.difficulty,
        terrain_task="flat",
        seed=args.seed,
        locomotion_model=args.locomotion_model,
    )
    assert environment._legged_loco is not None
    # Keep the original Go2 command speed and curvature untouched.  This
    # gain applies only to the learned 12-joint residual around the existing
    # physical trot prior; reducing it on sharp high-speed turns prevents a
    # residual overshoot without any root force or kinematic translation.
    environment._legged_loco.deployment_action_gain = float(args.go2_policy_action_gain)
    environment._legged_loco.observation_action_gain = float(args.go2_policy_action_gain)
    # This deployment scale is applied before the learned low-level PPO sees
    # its body-velocity command.  It preserves the requested easy/medium/hard
    # ordering and curved-route complexity, while keeping the real PX4 camera
    # reacquisition case inside the finite recorded course.  It is not a
    # kinematic base translation: the 12-joint PPO and sole contacts still
    # generate every metre of Go2 movement.
    environment.profile = dict(environment.profile)
    environment.profile["path_speed"] *= float(args.go2_speed_scale)
    environment.profile["turn_angle_rad"] *= float(args.go2_turn_scale)
    # Mission-only camera gate; it is not an RL input or a physical-contact
    # measurement.  Store it on the environment to keep the mission bridge
    # free of hard-coded difficulty values.
    environment.capture_radius_m = float(args.capture_radius_m)
    # An optional acquisition interval may precede the continuous route.
    # Shift heading and heading-rate together so the foot-contact gait
    # receives one self-consistent curved route.
    motion_delay_s = float(args.go2_motion_start_delay_s)
    if motion_delay_s > 0.0:
        path_command = environment._path_command
        path_heading = environment._path_heading
        path_heading_rate = environment._path_heading_rate

        def delayed_path_command(time_s: float) -> np.ndarray:
            return (
                np.zeros(3, dtype=np.float64)
                if time_s < motion_delay_s
                else path_command(time_s - motion_delay_s)
            )

        def delayed_path_heading(time_s: float) -> float:
            return 0.0 if time_s < motion_delay_s else float(path_heading(time_s - motion_delay_s))

        def delayed_path_heading_rate(time_s: float) -> float:
            return 0.0 if time_s < motion_delay_s else float(path_heading_rate(time_s - motion_delay_s))

        environment._path_command = delayed_path_command
        environment._path_heading = delayed_path_heading
        environment._path_heading_rate = delayed_path_heading_rate
    observation, reset_info = environment.reset(seed=args.seed)
    # ``reset`` builds its generic mission altitude.  Apply the deployment
    # value after reset so the PX4 companion really keeps the target in the
    # useful down-camera envelope instead of silently reverting to 2.72 m.
    environment._search_altitude = float(args.search_altitude_world_m)
    # The real PX4 local origin is anchored at the MuJoCo position that
    # generated the first HIL sample.  This is the only world/local mapping.
    start_world_enu_m = environment._sensor("drone_gps_position")
    propeller_ids = np.array(
        [
            mujoco.mj_name2id(environment.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in (
                "propeller_front_right", "propeller_rear_left",
                "propeller_front_left", "propeller_rear_right",
            )
        ],
        dtype=np.int32,
    )
    propeller_base_quaternions = environment.model.geom_quat[propeller_ids].copy()
    last_frame: np.ndarray | None = None
    next_frame_time = 0.0
    render_states: list[tuple[float, np.ndarray, np.ndarray]] = []
    landed = False
    terminal_info: dict[str, float] = {}
    mission: Px4VisionMission | None = None

    with Px4MujocoHilSession(log_path=args.px4_log_file) as session:
        hil_time_base_us = warm_and_arm_px4(session, environment)
        mission = Px4VisionMission(
            environment,
            session,
            start_world_enu_m=start_world_enu_m,
            hil_time_base_us=hil_time_base_us,
            flight_policy_residual_gain=float(args.flight_policy_residual_gain),
        )
        mission.update_ekf_estimate(force=True)
        environment._update_onboard_estimator = MethodType(
            lambda _self, *, force=False: mission.update_ekf_estimate(force=force), environment
        )
        environment._drone_control = MethodType(
            lambda _self, action, *, update_alignment: mission.apply_px4_control(
                action, update_alignment=update_alignment
            ),
            environment,
        )
        # PX4's own IMU/EKF/commander performs flight-state handling.  The
        # former MuJoCo-only direct-wrench impact retry assumes a commanded
        # body force, so it must not interpret the PX4 motor output.
        environment._update_imu_landing_state = MethodType(lambda _self: None, environment)
        environment._update_qr_camera_measurement(force=True)
        observation = environment._observation()

        with args.trace_file.open("w", newline="", encoding="utf-8") as trace_handle:
            trace = csv.writer(trace_handle)
            trace.writerow(
                (
                    "sim_time_s", "qr_error_m", "relative_altitude_m", "qr_detected",
                    "px4_xy_innovation", "px4_z_innovation", "px4_motor_mean", "px4_armed",
                    "px4_offboard", "desired_vx_mps", "desired_vy_mps", "desired_vz_mps",
                    "physical_x_m", "physical_y_m", "physical_z_m",
                    "ekf_x_m", "ekf_y_m", "ekf_z_m",
                    "success", "skid_contacts", "landing_normal_force_n",
                )
            )
            for _ in range(min(args.max_steps, environment.max_steps)):
                action = policy.run(
                    [policy_output], {policy_input: observation[np.newaxis, :].astype(np.float32)}
                )[0][0]
                action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
                observation, _reward, terminated, truncated, info = environment.step(action)
                terminal_info = info
                while environment.data.time + 1.0e-9 >= next_frame_time:
                    # Rendering can take longer than PX4's Offboard-loss
                    # window on a software EGL device.  Snapshot the exact
                    # MuJoCo state now and render it only after the HIL run;
                    # no live camera work is allowed to interrupt MAVLink
                    # sensor/setpoint streaming.
                    render_states.append(
                        (
                            float(environment.data.time),
                            environment.data.qpos.copy(),
                            environment.data.qvel.copy(),
                        )
                    )
                    next_frame_time += 1.0 / float(args.fps)
                trace.writerow(
                    (
                        f"{environment.data.time:.3f}",
                        f"{np.linalg.norm(environment._horizontal_error()):.6f}",
                        f"{environment._relative_altitude():.6f}",
                        int(environment._qr_detected),
                        f"{mission.horizontal_innovation_ratio:.8f}",
                        f"{mission.vertical_innovation_ratio:.8f}",
                        f"{np.mean(mission.last_motor_outputs):.6f}",
                        int(session.armed),
                        int(session.offboard_active),
                        *(f"{value:.6f}" for value in mission.last_desired_world_velocity),
                        *(f"{value:.6f}" for value in environment.drone_position),
                        *(f"{value:.6f}" for value in environment._estimated_position),
                        int(info["success"] > 0.5),
                        int(info["offline_sim_landing_skid_contacts"]),
                        f"{info['offline_sim_landing_normal_force_n']:.6f}",
                    )
                )
                if terminated or truncated:
                    landed = bool(info["success"] > 0.5)
                    break
            if not render_states:
                render_states.append(
                    (
                        float(environment.data.time),
                        environment.data.qpos.copy(),
                        environment.data.qvel.copy(),
                    )
                )
        metrics = {
            "backend": "mujoco_flat_x500_with_external_px4_sitl_ekf2_hil",
            "terrain": "flat",
            "algorithm": f"{args.algorithm.upper()} ONNX residual policy",
            "seed": args.seed,
            "difficulty": args.difficulty,
            "flight_policy": {
                "onnx_residual_gain_mps": float(args.flight_policy_residual_gain),
                "role": "bounded horizontal trim; camera/PnP visual servo remains primary",
            },
            "go2_route": {
                "speed_scale": float(args.go2_speed_scale),
                "turn_scale": float(args.go2_turn_scale),
                "command_speed_mps": float(environment.profile["path_speed"]),
                "turn_angle_rad": float(environment.profile["turn_angle_rad"]),
                "turn_frequency_hz": float(environment.profile["turn_frequency_hz"]),
                "motion_start_delay_s": motion_delay_s,
                "capture_radius_m": float(args.capture_radius_m),
                "search_altitude_world_m": float(args.search_altitude_world_m),
            },
            "locomotion": {
                "model": str(args.locomotion_model),
                "sha256": sha256_file(args.locomotion_model),
            "mode": "learned 12-joint Go2 PPO; foot-contact locomotion only",
            "policy_action_gain": float(args.go2_policy_action_gain),
            },
            "success": landed,
            "reset": reset_info,
            "terminal": terminal_info,
            "px4": {
                "armed": session.armed,
                "offboard_active": session.offboard_active,
                "command_ack_results": session.diagnostics.command_ack_results,
                "hil_sensor_messages": session.diagnostics.hil_sensor_messages,
                "hil_gps_messages": session.diagnostics.hil_gps_messages,
                "hil_sensor_time_usec": session.diagnostics.last_hil_sensor_time_usec,
                "hil_gps_time_usec": session.diagnostics.last_hil_gps_time_usec,
                "hil_time_regressions": session.diagnostics.hil_time_regressions,
                "offboard_velocity_messages": session.diagnostics.offboard_velocity_messages,
                "offboard_send_failures": session.diagnostics.offboard_send_failures,
                "last_offboard_target": session.diagnostics.last_offboard_target,
                "target_system": session.target_system,
                "target_component": session.target_component,
                "actuator_messages": session.diagnostics.actuator_messages,
                "armed_actuator_messages": session.diagnostics.actuator_armed_messages,
                "ekf_xy_innovation_ratio": mission.horizontal_innovation_ratio,
                "ekf_z_innovation_ratio": mission.vertical_innovation_ratio,
                "ekf_solution_status_flags": session.ekf.solution_status_flags,
                # Copied once the isolated PX4 rootfs closes below.
                "ulog_file": None,
            },
        }
    if mission is None:
        raise RuntimeError("PX4 HIL inference did not initialize")
    metrics["px4"]["ulog_file"] = str(session.ulog_path) if session.ulog_path is not None else None
    renderer = mujoco.Renderer(environment.model, height=720, width=1280)
    down_camera_option = mujoco.MjvOption()
    down_camera_option.geomgroup[1] = 0
    with imageio.get_writer(
        args.video_file,
        fps=args.fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=1,
    ) as writer:
        for frame_time, qpos, qvel in render_states:
            environment.data.time = frame_time
            environment.data.qpos[:] = qpos
            environment.data.qvel[:] = qvel
            mujoco.mj_forward(environment.model, environment.data)
            last_frame = render_dual_view(
                environment,
                renderer=renderer,
                down_camera_option=down_camera_option,
                propeller_ids=propeller_ids,
                propeller_base_quaternions=propeller_base_quaternions,
                mission=mission,
                algorithm=args.algorithm,
            )
            writer.append_data(last_frame)
    renderer.close()
    if last_frame is None:
        raise RuntimeError("PX4 HIL inference has no saved rendering state")
    imageio.imwrite(args.snapshot_file, last_frame)
    args.metrics_file.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    environment.close()
    if not landed:
        raise SystemExit(f"PX4 HIL flat {args.algorithm.upper()} run finished without a stable QR-deck landing")
    print(f"PX4 HIL flat {args.algorithm.upper()} landing confirmed: {args.video_file}")


if __name__ == "__main__":
    main()
