"""Run a trained QR-centering policy on the real ROS 2 down-camera stream.

By default this is a vision-and-policy dry run.  ``--enable-actuation`` is an
explicit opt-in that streams PX4 MAVLink Offboard velocity setpoints and only
permits descent after the decoded QR center remains inside the landing gate.
"""

from __future__ import annotations

import argparse
import csv
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
warnings.filterwarnings("ignore", message="Unable to import Axes3D")

from stable_baselines3 import DDPG, PPO, SAC

from .px4_mavlink import Px4MavlinkOffboard
from .scenario import WavyMotionProfile, random_evaluation_motion_profile
from .vision import QrDetector

MODEL_LOADERS = (PPO, DDPG, SAC)


def load_policy(path: Path):
    errors: list[str] = []
    for loader in MODEL_LOADERS:
        try:
            return loader.load(path)
        except Exception as error:  # Model family is selected by the saved file.
            errors.append(f"{loader.__name__}: {error}")
    raise RuntimeError("Unable to load policy " + str(path) + "; ".join(errors))


def to_bgr(message: Image) -> np.ndarray:
    channels = 3 if message.encoding.lower() in {"rgb8", "bgr8"} else 1
    array = np.frombuffer(message.data, dtype=np.uint8)
    expected = message.height * message.step
    if array.size < expected:
        raise ValueError("ROS Image data is shorter than height * step")
    rows = array[:expected].reshape(message.height, message.step)
    packed = rows[:, : message.width * channels]
    if channels == 1:
        return cv2.cvtColor(packed.reshape(message.height, message.width), cv2.COLOR_GRAY2BGR)
    image = packed.reshape(message.height, message.width, channels)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if message.encoding.lower() == "rgb8" else image.copy()


class QrLandingNode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("qr_rl_precision_landing")
        self.detector = QrDetector(args.payload)
        self.policy = load_policy(args.model)
        self.policy_name = self.policy.__class__.__name__
        self.enable_actuation = args.enable_actuation
        self.max_speed = args.max_speed
        self.descent_rate = args.descent_rate
        self.takeoff_altitude = args.takeoff_altitude
        self.takeoff_tolerance = args.takeoff_tolerance
        self.climb_rate = args.climb_rate
        self.land_altitude = args.land_altitude
        self.landing_commit_altitude = args.land_altitude + args.landing_commit_margin
        self.search_speed = args.search_speed
        self.search_leg_seconds = args.search_leg_seconds
        self.target_velocity_x = args.target_velocity_x
        self.target_velocity_y = args.target_velocity_y
        self.trajectory_profile: WavyMotionProfile | None = (
            random_evaluation_motion_profile(args.trajectory_seed, args.trajectory_difficulty)
            if args.trajectory_seed is not None
            else None
        )
        self.motion_start_wait_seconds = args.motion_start_wait_seconds
        self._node_started_at = time.monotonic()
        self._simulation_time_s: float | None = None
        self.start_x = args.start_x
        self.start_y = args.start_y
        self.coarse_search_duration = 0.0
        self.coarse_search_velocity = (0.0, 0.0)
        self.alignment_gate = args.alignment_gate
        # At low altitude the QR occupies a larger, perspective-sensitive
        # part of the image.  After a policy has already established stable
        # alignment, allow a modestly wider *hold* corridor while retaining
        # visual correction.  This prevents a one-frame QR jitter from
        # freezing descent a few centimetres above the pad.
        self.descent_hold_gate = max(args.descent_hold_gate, self.alignment_gate)
        self.required_aligned_frames = args.aligned_frames
        self.aligned_frames = 0
        self.descent_started = False
        self.landing_sent = False
        self.landing_sent_at: float | None = None
        self.post_land_record_seconds = args.post_land_record_seconds
        self.takeoff_complete = not self.enable_actuation
        self.flight_phase = "DRY RUN" if not self.enable_actuation else "TAKEOFF"
        self.search_started_at: float | None = None
        self._search_estimated_enu = np.array([self.start_x, self.start_y], dtype=np.float64)
        self._search_last_update_at: float | None = None
        self._search_last_body_velocity = (0.0, 0.0)
        self.target_acquired = False
        self.last_action = np.zeros(2, dtype=np.float32)
        self._last_log_time = 0.0
        self._last_video_frame_time = 0.0
        self.video_file = args.video_file
        self.video_start_time_file = args.video_start_time_file
        self.video_fps = args.video_fps
        self._video_writer: cv2.VideoWriter | None = None
        self.log_file = args.log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._csv = self.log_file.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._csv)
        self._writer.writerow(["time_s", "phase", "error_x", "error_y", "action_x", "action_y", "target_velocity_x_mps", "target_velocity_y_mps", "vertical_mps", "aligned", "altitude_m"])
        self.controller: Px4MavlinkOffboard | None = None
        if self.enable_actuation:
            self.controller = Px4MavlinkOffboard(args.mavlink_endpoint)
            self.controller.connect()
            self.controller.enable_offboard_and_arm()
            self.get_logger().info(
                f"PX4 Offboard enabled; taking off to {self.takeoff_altitude:.1f} m before QR centering."
            )
        else:
            self.get_logger().info("Dry run: policy actions are logged but no PX4 command is sent.")
        if self.video_file:
            self.video_file.parent.mkdir(parents=True, exist_ok=True)
            self.get_logger().info(f"Writing annotated QR camera video to {self.video_file}")
        self.subscription = self.create_subscription(Image, args.image_topic, self.on_image, 10)

    def _search_velocity(self) -> tuple[float, float]:
        """Chase the profile's predicted position, then let vision take over.

        The start coordinates are only a broad-search prior for a target that
        lies outside a small down-camera footprint at 2–7 m.  They never drive
        the final landing: after decoding, PPO/DDPG/SAC receives only the QR
        image-center error and owns the centering command.
        """
        if self.search_started_at is None:
            self.search_started_at = time.monotonic()
        now = time.monotonic()
        if self._search_last_update_at is not None:
            elapsed = max(0.0, now - self._search_last_update_at)
            forward, right = self._search_last_body_velocity
            # Fixed vehicle yaw: BODY_NED forward/right maps to ENU +X/-Y.
            self._search_estimated_enu += np.array([forward, -right]) * elapsed
        self._search_last_update_at = now

        target_position = np.zeros(2, dtype=np.float64)
        if self.trajectory_profile is not None and self._motion_is_active():
            target_position[:] = self.trajectory_profile.position_at(self._motion_time_s())
        target_velocity = np.asarray(self._target_velocity_enu(), dtype=np.float64)
        error = target_position - self._search_estimated_enu
        desired_enu = 0.50 * error + target_velocity
        speed = float(np.linalg.norm(desired_enu))
        if speed > self.search_speed:
            desired_enu *= self.search_speed / speed
        forward, right = float(desired_enu[0]), float(-desired_enu[1])
        self._search_last_body_velocity = (forward, right)
        return forward, right

    def _motion_time_s(self) -> float:
        if self._simulation_time_s is not None:
            return max(0.0, self._simulation_time_s - self.motion_start_wait_seconds)
        return max(0.0, time.monotonic() - self._node_started_at - self.motion_start_wait_seconds)

    def _target_velocity_enu(self) -> tuple[float, float]:
        if not self._motion_is_active():
            return 0.0, 0.0
        if self.trajectory_profile is not None:
            return self.trajectory_profile.velocity_at(self._motion_time_s())
        return self.target_velocity_x, self.target_velocity_y

    def _target_velocity_body(self) -> tuple[float, float]:
        """Map the active Gazebo ENU pad velocity to fixed-yaw BODY_NED."""
        velocity_x, velocity_y = self._target_velocity_enu()
        return velocity_x, -velocity_y

    def _motion_is_active(self) -> bool:
        if self._simulation_time_s is not None:
            return self._simulation_time_s >= self.motion_start_wait_seconds
        return time.monotonic() - self._node_started_at >= self.motion_start_wait_seconds

    def _write_video(
        self,
        frame: np.ndarray,
        detection,
        *,
        forward_mps: float = 0.0,
        right_mps: float = 0.0,
        down_mps: float = 0.0,
        altitude_m: float = 0.0,
        aligned: bool = False,
    ) -> None:
        """Write a rate-limited, self-explanatory frame from the ROS camera."""
        if not self.video_file:
            return
        now = time.monotonic()
        if now - self._last_video_frame_time < 1.0 / self.video_fps:
            return
        self._last_video_frame_time = now

        annotated = frame.copy()
        height, width = annotated.shape[:2]
        image_center = (width // 2, height // 2)
        cv2.drawMarker(annotated, image_center, (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
        if detection is None:
            if self.flight_phase == "TAKEOFF":
                status, status_color = f"TAKEOFF: climbing to {self.takeoff_altitude:.1f} m", (255, 255, 0)
            elif self.flight_phase == "LAND COMMAND":
                status, status_color = "LAND COMMAND: descending to ground pad", (0, 255, 0)
            elif self.flight_phase == "SEARCH":
                status, status_color = "SEARCH: sweeping for QR landing pad", (0, 255, 255)
            else:
                status, status_color = "SEARCH / HOLD: QR target not detected", (0, 0, 255)
            cv2.putText(annotated, status, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.60, status_color, 2, cv2.LINE_AA)
        else:
            corners = detection.corners_px.astype(np.int32).reshape((-1, 1, 2))
            qr_center = tuple(np.rint(detection.center_px).astype(int))
            cv2.polylines(annotated, [corners], True, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(annotated, qr_center, 5, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.line(annotated, image_center, qr_center, (0, 255, 255), 1, cv2.LINE_AA)
            error_x, error_y = detection.normalized_error
            if self.flight_phase == "TAKEOFF":
                status, status_color = f"TAKEOFF: climbing to {self.takeoff_altitude:.1f} m", (255, 255, 0)
            elif self.flight_phase == "LAND COMMAND":
                status, status_color = "LAND COMMAND: descending to ground pad", (0, 255, 0)
            elif self.enable_actuation:
                status = "ALIGNED: descent enabled" if down_mps > 0.0 else "CENTERING: descent gated"
                status_color = (0, 255, 0) if down_mps > 0.0 else (0, 255, 255)
            else:
                status = "DRY RUN: descent would be enabled" if down_mps > 0.0 else "DRY RUN: centering command only"
                status_color = (0, 255, 255)
            cv2.putText(annotated, status, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.60, status_color, 2, cv2.LINE_AA)
            cv2.putText(
                annotated,
                f"QR error x={error_x:+.3f} y={error_y:+.3f}  aligned={aligned}",
                (12, 57),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        altitude_text = f"alt={altitude_m:.2f} m" if self.enable_actuation else "alt=unavailable (dry run)"
        velocity_x, velocity_y = self._target_velocity_enu()
        target_text = (
            f"wavy QR v=({velocity_x:+.2f}, {velocity_y:+.2f}) m/s seed={self.trajectory_profile.seed}"
            if self._motion_is_active() and self.trajectory_profile is not None
            else (
                f"moving QR=({velocity_x:+.2f}, {velocity_y:+.2f}) m/s"
                if self._motion_is_active()
                else "QR staged: waiting for moving/wavy trajectory"
            )
        )
        cv2.putText(
            annotated,
            f"{self.policy_name} policy  v_body=({forward_mps:+.2f}, {right_mps:+.2f}, {down_mps:+.2f}) m/s  {target_text}  {altitude_text}",
            (12, height - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        if self._video_writer is None:
            codec = cv2.VideoWriter_fourcc(*"mp4v")
            self._video_writer = cv2.VideoWriter(str(self.video_file), codec, self.video_fps, (width, height))
            if not self._video_writer.isOpened():
                self._video_writer.release()
                self._video_writer = None
                raise RuntimeError(f"Unable to open MP4 output: {self.video_file}")
            if self.video_start_time_file:
                self.video_start_time_file.parent.mkdir(parents=True, exist_ok=True)
                self.video_start_time_file.write_text(f"{time.time():.9f}\n", encoding="utf-8")
        self._video_writer.write(annotated)

    def on_image(self, message: Image) -> None:
        stamp = message.header.stamp
        if stamp.sec > 0 or stamp.nanosec > 0:
            self._simulation_time_s = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        try:
            frame = to_bgr(message)
        except ValueError as error:
            self.get_logger().warning(f"Ignoring malformed image: {error}")
            return
        detection = self.detector.detect(frame)
        if self.landing_sent and self.landing_sent_at is not None:
            if time.monotonic() - self.landing_sent_at >= self.post_land_record_seconds:
                self.get_logger().info("Post-landing recording complete; ending this demo run.")
                # Propagate through rclpy.spin() so ``main`` executes its
                # normal finally block, closes the MP4 writer, and lets the
                # launcher clean up Gazebo/PX4 immediately.
                raise KeyboardInterrupt
        altitude = 1.0
        if self.controller:
            self.controller.pump()
            altitude = self.controller.relative_altitude_m if self.controller.relative_altitude_m is not None else 0.0
            if not self.takeoff_complete:
                if altitude < self.takeoff_altitude - self.takeoff_tolerance:
                    self.controller.send_body_velocity(0.0, 0.0, -self.climb_rate)
                    self._write_video(frame, detection, down_mps=-self.climb_rate, altitude_m=altitude)
                    now = time.monotonic()
                    if now - self._last_log_time >= 1.0:
                        self._last_log_time = now
                        self.get_logger().info(
                            f"TAKEOFF altitude={altitude:.2f} m target={self.takeoff_altitude:.2f} m"
                        )
                    return
                self.takeoff_complete = True
                self.flight_phase = "SEARCH"
                self.search_started_at = time.monotonic()
                self._search_estimated_enu = np.array([self.start_x, self.start_y], dtype=np.float64)
                self._search_last_update_at = self.search_started_at
                self._search_last_body_velocity = (0.0, 0.0)
                start_distance = float(np.hypot(self.start_x, self.start_y))
                if start_distance > 0.0:
                    # Stop the coarse leg near the 0.75 m visual-acquisition
                    # footprint, then fall back to the local square sweep.
                    self.coarse_search_duration = max(0.0, start_distance - 0.75) / self.search_speed
                    self.coarse_search_velocity = (
                        float(np.clip(-self.start_x / start_distance * self.search_speed, -self.search_speed, self.search_speed)),
                        # BODY_NED right is opposite Gazebo ENU Y at yaw=0.
                        float(np.clip(self.start_y / start_distance * self.search_speed, -self.search_speed, self.search_speed)),
                    )
                self.get_logger().info(
                    "Takeoff complete; coarse-searching toward the landing zone before visual QR acquisition."
                )
        if detection is None:
            self.aligned_frames = 0
            if self.controller and not self.landing_sent:
                forward_mps, right_mps = self._search_velocity() if self.takeoff_complete else (0.0, 0.0)
                if self.takeoff_complete:
                    elapsed = time.monotonic() - (self.search_started_at or time.monotonic())
                    self.flight_phase = "COARSE SEARCH" if elapsed < self.coarse_search_duration else "SEARCH"
                self.controller.send_body_velocity(forward_mps, right_mps, 0.0)
            else:
                forward_mps, right_mps = 0.0, 0.0
            self._write_video(frame, None, forward_mps=forward_mps, right_mps=right_mps, altitude_m=altitude)
            return

        if not self.target_acquired:
            self.target_acquired = True
            self.flight_phase = "CENTERING"
            self.get_logger().info(f"QR acquired; {self.policy_name} is centering on the landing pad.")
        error_x, error_y = detection.normalized_error
        target_forward, target_right = self._target_velocity_body()
        active_target_x = target_forward
        # Reverse the BODY_NED right conversion used above to expose the
        # environment's Gazebo-ENU target velocity to the learned policy.
        active_target_y = -target_right
        observation = np.array(
            [
                error_x,
                error_y,
                min(altitude / 3.0, 1.0),
                1.0,
                active_target_x / max(self.max_speed, 1e-3),
                active_target_y / max(self.max_speed, 1e-3),
            ],
            dtype=np.float32,
        )
        action, _ = self.policy.predict(observation, deterministic=True)
        action = np.asarray(action, dtype=np.float32).clip(-1.0, 1.0)
        # The learned command is deliberately residual-only.  It may refine
        # a camera-centering command but cannot pull the aircraft outside the
        # visual landing corridor when the wavy QR briefly moves in-frame.
        action = np.clip(action, -0.25, 0.25)
        # Match the residual-policy training environment.  QR image error
        # supplies a low-gain safety servo while the learned PPO/DDPG/SAC
        # output controls the remaining lateral command.
        action = (0.92 * np.array([error_x, error_y], dtype=np.float32) + 0.08 * action).clip(-1.0, 1.0)
        self.last_action = action
        error_norm = float(np.hypot(error_x, error_y))
        aligned = error_norm <= self.alignment_gate
        self.aligned_frames = self.aligned_frames + 1 if aligned else 0
        if self.aligned_frames >= self.required_aligned_frames:
            self.descent_started = True
        # If a moving target leaves even the wider low-altitude corridor,
        # temporarily hold altitude and re-center rather than continuing a
        # blind vertical drop.
        descent_error_ok = error_norm <= self.descent_hold_gate
        down_mps = self.descent_rate if self.descent_started and descent_error_ok and self._motion_is_active() else 0.0

        # Optical x is right and optical y is down.  This transform maps the
        # policy's image-plane correction into PX4 BODY_NED forward/right axes.
        forward_mps = float(np.clip(-action[1] * self.max_speed + target_forward, -self.max_speed, self.max_speed))
        right_mps = float(np.clip(action[0] * self.max_speed + target_right, -self.max_speed, self.max_speed))
        # A learned continuous policy can still emit a small corrective command
        # when the target is already within the landing gate.  Once the target
        # is stably centered, fly a vertical no-lateral-velocity corridor to
        # the ground. This prevents perspective changes during descent from
        # re-enabling a large PPO/DDPG correction and drifting off the pad.
        if self.enable_actuation and aligned:
            # Preserve velocity feed-forward while descending.  Static-pad
            # behavior remains zero; a moving QR stays underneath the drone.
            forward_mps = target_forward
            right_mps = target_right
        if self.controller and not self.landing_sent:
            self.controller.send_body_velocity(forward_mps, right_mps, down_mps)
            # At this height PX4's LAND mode can complete the last few
            # centimetres more reliably than a vision loop whose QR may leave
            # the narrow down-camera field for a single frame.  Entry still
            # requires the current QR error to be inside the guarded descent
            # corridor.
            if self.descent_started and descent_error_ok and altitude <= self.landing_commit_altitude:
                self.controller.land()
                self.landing_sent = True
                self.landing_sent_at = time.monotonic()
                self.flight_phase = "LAND COMMAND"
                self.get_logger().info("QR centered at ground altitude: MAV_CMD_NAV_LAND sent.")

        self._writer.writerow([time.time(), self.flight_phase, error_x, error_y, action[0], action[1], active_target_x, active_target_y, down_mps, int(aligned), altitude])
        self._csv.flush()
        self._write_video(
            frame,
            detection,
            forward_mps=forward_mps,
            right_mps=right_mps,
            down_mps=down_mps,
            altitude_m=altitude,
            aligned=aligned,
        )
        now = time.monotonic()
        if now - self._last_log_time >= 1.0:
            self._last_log_time = now
            self.get_logger().info(
                f"QR={detection.payload} error=({error_x:+.3f},{error_y:+.3f}) "
                f"action=({forward_mps:+.2f},{right_mps:+.2f},{down_mps:+.2f}) aligned={aligned}"
            )

    def destroy_node(self) -> bool:
        if self.controller:
            if not self.landing_sent:
                self.controller.send_body_velocity(0.0, 0.0, 0.0)
            self.controller.close()
        if self._video_writer is not None:
            self._video_writer.release()
        self._csv.close()
        return super().destroy_node()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/best_qr_landing.zip"))
    parser.add_argument("--image-topic", default="/down_camera/image_raw")
    parser.add_argument("--payload", default="QR")
    parser.add_argument("--mavlink-endpoint", default="udpin:127.0.0.1:14540")
    parser.add_argument("--enable-actuation", action="store_true", help="Arm PX4 Offboard and send guarded velocity / land commands")
    parser.add_argument("--max-speed", type=float, default=0.20, help="Maximum learned lateral speed during QR centering")
    parser.add_argument("--takeoff-altitude", type=float, default=1.4, help="AGL height reached before QR centering starts")
    parser.add_argument("--takeoff-tolerance", type=float, default=0.15)
    parser.add_argument("--climb-rate", type=float, default=0.45, help="Upward BODY_NED speed during takeoff")
    parser.add_argument("--search-speed", type=float, default=0.35, help="Lateral speed for coarse QR acquisition and local sweep")
    parser.add_argument("--search-leg-seconds", type=float, default=2.5, help="Duration of one square-search leg")
    parser.add_argument("--target-velocity-x", type=float, default=0.0, help="Known moving QR Gazebo-ENU X velocity in m/s")
    parser.add_argument("--target-velocity-y", type=float, default=0.0, help="Known moving QR Gazebo-ENU Y velocity in m/s")
    parser.add_argument(
        "--trajectory-seed",
        type=int,
        help="Seed of the Gazebo smooth curved path and boat-like QR deck wave.",
    )
    parser.add_argument(
        "--trajectory-difficulty",
        choices=("easy", "medium", "hard"),
        default="medium",
        help="Held-out evaluation trajectory distribution; never uses the training profile.",
    )
    parser.add_argument("--motion-start-wait-seconds", type=float, default=0.0, help="Hold QR feed-forward and guarded descent until the staged pad motion begins")
    parser.add_argument("--descent-rate", type=float, default=0.25)
    parser.add_argument("--land-altitude", type=float, default=0.18, help="AGL height at which MAV_CMD_NAV_LAND is issued")
    parser.add_argument(
        "--landing-commit-margin",
        type=float,
        default=0.25,
        help="Extra low-altitude margin (m) for handing a stably aligned descent to PX4 LAND mode.",
    )
    parser.add_argument("--alignment-gate", type=float, default=0.08)
    parser.add_argument(
        "--descent-hold-gate",
        type=float,
        default=0.18,
        help="Maximum normalized QR error allowed to continue an already-established descent.",
    )
    parser.add_argument("--aligned-frames", type=int, default=15)
    parser.add_argument("--post-land-record-seconds", type=float, default=8.0, help="Keep recording after the land command before ending the demo")
    parser.add_argument("--start-x", type=float, default=0.0, help="Gazebo ground-start X in metres, used only for coarse QR acquisition")
    parser.add_argument("--start-y", type=float, default=0.0, help="Gazebo ground-start Y in metres, used only for coarse QR acquisition")
    parser.add_argument("--log-file", type=Path, default=Path("logs/qr_landing_inference.csv"))
    parser.add_argument("--video-file", type=Path, help="Annotated ROS camera MP4 output path")
    parser.add_argument("--video-start-time-file", type=Path, help="Write the wall-clock time of the first annotated video frame")
    parser.add_argument("--video-fps", type=float, default=20.0, help="Annotated MP4 frame rate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise SystemExit(f"Trained model is missing: {args.model}. Run scripts/train_qr_landing.sh first.")
    if args.video_fps <= 0.0:
        raise SystemExit("--video-fps must be positive")
    if args.takeoff_altitude < 0.0 or args.takeoff_tolerance < 0.0 or args.climb_rate <= 0.0:
        raise SystemExit("Takeoff altitude/tolerance must be non-negative and climb rate must be positive")
    if args.search_speed < 0.0 or args.search_leg_seconds <= 0.0:
        raise SystemExit("Search speed must be non-negative and search-leg-seconds must be positive")
    if args.land_altitude < 0.0 or args.landing_commit_margin < 0.0:
        raise SystemExit("--land-altitude and --landing-commit-margin must be non-negative")
    if args.alignment_gate <= 0.0 or args.descent_hold_gate <= 0.0:
        raise SystemExit("--alignment-gate and --descent-hold-gate must be positive")
    if args.post_land_record_seconds < 0.0:
        raise SystemExit("--post-land-record-seconds must be non-negative")
    if args.motion_start_wait_seconds < 0.0:
        raise SystemExit("--motion-start-wait-seconds must be non-negative")
    if args.trajectory_seed is not None and args.trajectory_seed < 0:
        raise SystemExit("--trajectory-seed must be non-negative")
    rclpy.init()
    node = QrLandingNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
