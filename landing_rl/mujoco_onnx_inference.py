"""Run a landing policy through ONNX Runtime in MuJoCo and record a follow view."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

# Must be chosen before importing MuJoCo through the environment module.
os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np
import onnxruntime as ort

from .mujoco_environment import MujocoQrPrecisionLandingEnv


def follow_camera(environment: MujocoQrPrecisionLandingEnv) -> mujoco.MjvCamera:
    """Create a camera that follows the drone and keeps the QR ahead in view."""
    drone = environment.drone_position
    target = environment.pad_position
    direction = target[:2] - drone[:2]
    heading = float(np.degrees(np.arctan2(direction[1], direction[0]))) if np.linalg.norm(direction) > 0.05 else 0.0
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    # The tracked look-at position is recomputed for every rendered frame; it
    # is deliberately not a static world camera.
    camera.lookat[:] = 0.78 * drone + 0.22 * target
    # A close follow distance keeps the actual X500 mesh recognisable in the
    # third-person evidence rather than reducing it to a few pixels.
    camera.distance = 2.75
    camera.azimuth = heading + 142.0
    camera.elevation = -24.0
    return camera


def animate_propellers(
    environment: MujocoQrPrecisionLandingEnv,
    propeller_ids: np.ndarray,
    base_quaternions: np.ndarray,
) -> None:
    """Spin the four original X500 propeller meshes for recorded inference.

    This is deliberately a render-only animation: the learned controller and
    MuJoCo flight dynamics remain identical.  Adjacent rotors counter-rotate
    as on a quadcopter, while a non-frame-aliased rate keeps the rotation
    visible in the 10-fps MP4.
    """
    rotor_rate_hz = 11.3
    angle = 2.0 * np.pi * rotor_rate_hz * float(environment.data.time)
    for index, (geom_id, base) in enumerate(zip(propeller_ids, base_quaternions)):
        signed_angle = angle if index < 2 else -angle
        half_angle = signed_angle / 2.0
        spin = np.array([np.cos(half_angle), 0.0, 0.0, np.sin(half_angle)])
        # World/body-Z spin multiplied by the original mesh alignment.
        w0, x0, y0, z0 = spin
        w1, x1, y1, z1 = base
        environment.model.geom_quat[geom_id] = (
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        )
    # Geom orientations live on the model, so update the derived render poses
    # before either camera reads the shared MuJoCo simulation state.
    mujoco.mj_forward(environment.model, environment.data)


def draw_hud(frame: np.ndarray, *, elapsed: float, error: float, altitude: float, phase: str) -> np.ndarray:
    image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.rectangle(image, (20, 18), (515, 126), (7, 17, 29), -1)
    cv2.rectangle(image, (20, 18), (515, 126), (93, 215, 255), 2)
    lines = [
        "MuJoCo + ONNX Runtime  |  moving QR precision landing",
        f"t={elapsed:05.1f}s  phase={phase}  QR error={error:.3f} m",
        f"altitude={altitude:.2f} m  |  FOLLOW CAMERA: drone tracked",
    ]
    for index, line in enumerate(lines):
        cv2.putText(image, line, (36, 49 + 27 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (234, 246, 255), 1, cv2.LINE_AA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def draw_down_camera_hud(frame: np.ndarray, *, error: float, detected: bool) -> np.ndarray:
    """Annotate the attached down-facing camera without changing its pixels."""
    image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    height, width = image.shape[:2]
    cv2.rectangle(image, (14, 14), (width - 14, 78), (7, 17, 29), -1)
    cv2.rectangle(image, (14, 14), (width - 14, 78), (93, 215, 255), 2)
    state = "QR DETECTED" if detected else "SEARCHING QR"
    cv2.putText(image, "ATTACHED DOWN CAMERA | same MuJoCo frame", (28, 41), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (234, 246, 255), 1, cv2.LINE_AA)
    cv2.putText(image, f"{state} | error={error:.3f} m", (28, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (173, 230, 255), 1, cv2.LINE_AA)
    center = (width // 2, height // 2)
    cv2.drawMarker(image, center, (55, 235, 255), cv2.MARKER_CROSS, 28, 2, cv2.LINE_AA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def compose_live_dual_view(third_person: np.ndarray, down_camera: np.ndarray) -> np.ndarray:
    """Place two renders from *one* simulation state in one video frame.

    This deliberately happens before the H.264 writer; it is not an MP4
    post-processing stitch and neither view has a separate clock.
    """
    canvas = np.zeros((720, 1920, 3), dtype=np.uint8)
    canvas[:, :1280] = third_person
    canvas[90:630, 1280:1920] = down_camera
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-model", type=Path, required=True)
    parser.add_argument("--difficulty", choices=("easy", "medium", "hard"), default="medium")
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--video-file", type=Path, required=True)
    parser.add_argument("--snapshot-file", type=Path, required=True)
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=1_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.onnx_model.exists():
        raise SystemExit(f"Missing ONNX model: {args.onnx_model}")
    session = ort.InferenceSession(str(args.onnx_model), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    environment = MujocoQrPrecisionLandingEnv(seed=args.seed, difficulty=args.difficulty)
    observation, reset_info = environment.reset(seed=args.seed)
    propeller_ids = np.array(
        [
            mujoco.mj_name2id(environment.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in ("propeller_front_right", "propeller_rear_left", "propeller_front_left", "propeller_rear_right")
        ],
        dtype=np.int32,
    )
    propeller_base_quaternions = environment.model.geom_quat[propeller_ids].copy()
    args.video_file.parent.mkdir(parents=True, exist_ok=True)
    args.snapshot_file.parent.mkdir(parents=True, exist_ok=True)
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    # One renderer is intentionally reused for both views. Some EGL drivers
    # allocate only one valid offscreen colour target per context; rendering
    # both cameras sequentially through this renderer keeps the paired frame
    # reliable while still using the exact same MuJoCo state.
    renderer = mujoco.Renderer(environment.model, height=720, width=1280)
    # The optical camera is attached to the X500, but a real camera does not
    # image its own propellers/airframe.  Group 1 contains those self-geoms.
    down_camera_option = mujoco.MjvOption()
    down_camera_option.geomgroup[1] = 0
    landed = False
    last_frame: np.ndarray | None = None

    with args.log_file.open("w", newline="", encoding="utf-8") as log_handle, imageio.get_writer(
        args.video_file, fps=args.fps, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=1
    ) as writer:
        csv_writer = csv.writer(log_handle)
        csv_writer.writerow(["sim_time_s", "phase", "error_m", "altitude_m", "action_x", "action_y", "detected", "aligned_streak", "onnx_provider"])
        for _ in range(min(args.max_steps, environment.max_steps)):
            action = session.run([output_name], {input_name: observation[np.newaxis, :].astype(np.float32)})[0][0]
            action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
            observation, _, terminated, truncated, info = environment.step(action)
            error = float(info["horizontal_error_m"])
            altitude = float(info["altitude_m"])
            detected = bool(observation[3] > 0.5)
            phase = "search" if not detected else ("descent" if info["aligned_streak"] >= 5 else "visual-center")
            animate_propellers(environment, propeller_ids, propeller_base_quaternions)
            camera = follow_camera(environment)
            renderer.update_scene(environment.data, camera=camera)
            third_person = draw_hud(renderer.render().copy(), elapsed=float(environment.data.time), error=error, altitude=altitude, phase=phase)
            # This named camera is physically attached below the converted
            # Gazebo X500 base link and is rendered immediately from the same
            # MuJoCo data object as the left third-person view.
            renderer.update_scene(environment.data, camera="down_camera", scene_option=down_camera_option)
            down_raw = cv2.resize(renderer.render().copy(), (640, 540), interpolation=cv2.INTER_AREA)
            down_camera = draw_down_camera_hud(down_raw, error=error, detected=detected)
            frame = compose_live_dual_view(third_person, down_camera)
            writer.append_data(frame)
            last_frame = frame
            csv_writer.writerow([f"{environment.data.time:.3f}", phase, f"{error:.6f}", f"{altitude:.6f}", f"{action[0]:.6f}", f"{action[1]:.6f}", int(detected), int(info["aligned_streak"]), "CPUExecutionProvider"])
            if terminated or truncated:
                landed = bool(info["success"] > 0.5)
                break
    renderer.close()
    if last_frame is None:
        raise RuntimeError("MuJoCo inference did not render a frame")
    imageio.imwrite(args.snapshot_file, last_frame)
    environment.close()
    if not landed:
        raise SystemExit("MuJoCo ONNX inference finished without a successful QR landing")
    print(f"LAND confirmed: {args.video_file} | snapshot: {args.snapshot_file}")


if __name__ == "__main__":
    main()
