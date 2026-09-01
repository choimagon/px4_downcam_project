"""Run sensor-only Go2 QR landing ONNX inference and record a dual-view MP4.

The policy consumes the 30 Hz QR-camera cache and 50 Hz PX4-estimator cache.
MuJoCo contact, deck, and Go2 truth is logged only under ``offline_sim_*``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np
import onnxruntime as ort

from .go2_qr_environment import (
    X500_SKID_CENTER_BODY_Z_M,
    X500_SKID_HALF_SIZE_M,
    X500_VISUAL_SKID_BOTTOM_BODY_Z_M,
    Go2BackQrLandingEnv,
)
from .go2_terrain import TERRAIN_TASKS, terrain_height_at


SEGMENTATION_SAMPLE_INTERVAL_FRAMES = 30


def terrain_hud_label(task: str, rough_level: int | None) -> str:
    """ASCII label: OpenCV Hershey fonts cannot reliably draw Korean text."""
    if task == "slope_up":
        return "UPHILL 10pct"
    if task == "slope_down":
        return "DOWNHILL 10pct"
    if task == "rough":
        return f"ROUGH LEVEL {rough_level}"
    return "FLAT"


def follow_camera(environment: Go2BackQrLandingEnv) -> mujoco.MjvCamera:
    drone = environment.drone_position
    pad = environment.pad_position
    direction = pad[:2] - drone[:2]
    heading = float(np.degrees(np.arctan2(direction[1], direction[0]))) if np.linalg.norm(direction) > 0.05 else 0.0
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = 0.55 * drone + 0.30 * pad + 0.15 * environment.base_position
    # Close enough to resolve the four landing feet and dorsal bridge while
    # retaining the full Go2 gait in the same third-person frame.
    # Keep *both* vehicles in shot from the initial 2–7 m annulus, then
    # automatically zoom in to a readable 2.40 m close-up for touchdown.
    # The 3-D distance also covers the initial altitude separation.
    separation = float(np.linalg.norm(drone - pad))
    # Terrain courses can put a search-phase X500 far above/aside the QR
    # deck.  Expand only that wide-search view so the whole airframe remains
    # safely inside frame; the 2.72 m touchdown minimum keeps the requested
    # close third-person landing view once it has approached the pad.
    if environment.terrain_task != "flat":
        camera.distance = max(2.72, 1.10 * separation + 1.75)
    else:
        camera.distance = max(2.72, 0.95 * separation + 1.57)
    camera.azimuth = heading + 145.0
    # Lower oblique angle makes the diagonal foot sequence readable; the
    # distance rule above still expands automatically to keep both vehicles
    # visible from the full 2--7 m starting annulus.
    camera.elevation = -20.0
    return camera


def render_state_fingerprint(environment: Go2BackQrLandingEnv) -> bytes:
    """Fingerprint the exact MuJoCo state consumed by both video views.

    Include derived transforms because they also contain the render-only
    propeller animation applied immediately before capture. Rendering must not
    advance physics or re-run kinematics between the synchronized views.
    """
    digest = hashlib.blake2b(digest_size=16)
    for value in (
        np.asarray([environment.data.time], dtype=np.float64),
        environment.data.qpos,
        environment.data.qvel,
        environment.data.xpos,
        environment.data.xmat,
        environment.data.geom_xpos,
        environment.data.geom_xmat,
    ):
        digest.update(np.ascontiguousarray(value).view(np.uint8))
    return digest.digest()


def assert_render_state_unchanged(
    environment: Go2BackQrLandingEnv,
    expected: bytes,
    *,
    rendered_view: str,
) -> None:
    """Fail immediately if a view was not rendered from the captured state."""
    if render_state_fingerprint(environment) != expected:
        raise RuntimeError(
            f"{rendered_view} render mutated MuJoCo state; dual-view frames are not synchronized"
        )


def drone_render_geom_ids(environment: Go2BackQrLandingEnv) -> np.ndarray:
    """Return every visible geom belonging to the X500 body subtree."""
    selected: list[int] = []
    for geom_id, initial_body_id in enumerate(environment.model.geom_bodyid):
        body_id = int(initial_body_id)
        while body_id > 0 and body_id != environment.drone_id:
            body_id = int(environment.model.body_parentid[body_id])
        if body_id == environment.drone_id:
            selected.append(geom_id)
    if not selected:
        raise RuntimeError("X500 body subtree contains no renderable geoms")
    return np.asarray(selected, dtype=np.int32)


def assert_drone_visible_in_segmentation(
    segmentation: np.ndarray,
    *,
    drone_geom_ids: np.ndarray,
    minimum_pixels: int = 200,
    viewport_margin_fraction: float = 0.03,
) -> int:
    """Require a resolved X500 silhouette safely inside the third-person view.

    MuJoCo segmentation pixels contain ``(object_id, object_type)``. Counting
    all geoms in the drone subtree is robust to colour and lighting, while the
    silhouette bounds catch a cropped aircraft even if its centre is visible.
    """
    if segmentation.ndim != 3 or segmentation.shape[2] != 2:
        raise RuntimeError(f"unexpected MuJoCo segmentation shape: {segmentation.shape}")
    height, width = segmentation.shape[:2]
    mask = (
        (segmentation[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
        & np.isin(segmentation[:, :, 0], drone_geom_ids)
    )
    pixel_count = int(np.count_nonzero(mask))
    if pixel_count < minimum_pixels:
        raise RuntimeError(
            f"third-person X500 visibility failed: {pixel_count} segmentation pixels "
            f"(minimum {minimum_pixels})"
        )
    rows, columns = np.nonzero(mask)
    margin_x = max(2, int(round(width * viewport_margin_fraction)))
    margin_y = max(2, int(round(height * viewport_margin_fraction)))
    if (
        int(columns.min()) < margin_x
        or int(columns.max()) >= width - margin_x
        or int(rows.min()) < margin_y
        or int(rows.max()) >= height - margin_y
    ):
        raise RuntimeError(
            "third-person X500 visibility failed: silhouette touches the "
            f"viewport safety margin ({margin_x}px horizontal, {margin_y}px vertical)"
        )
    return pixel_count


def drone_segmentation_box(
    segmentation: np.ndarray,
    *,
    drone_geom_ids: np.ndarray,
) -> tuple[float, float, float, float]:
    """Return the X500 segmentation centre and box size in render pixels."""
    mask = (
        (segmentation[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
        & np.isin(segmentation[:, :, 0], drone_geom_ids)
    )
    rows, columns = np.nonzero(mask)
    if not len(rows):
        raise RuntimeError("X500 segmentation contains no silhouette")
    return (
        0.5 * float(columns.min() + columns.max()),
        0.5 * float(rows.min() + rows.max()),
        float(columns.max() - columns.min() + 1),
        float(rows.max() - rows.min() + 1),
    )


def project_world_point(
    renderer: mujoco.Renderer,
    point_world: np.ndarray,
    *,
    frame_width: int,
    frame_height: int,
) -> tuple[float, float, float]:
    """Project a world point with MuJoCo's active perspective GL camera."""
    left_camera, right_camera = renderer.scene.camera
    position = 0.5 * (np.asarray(left_camera.pos) + np.asarray(right_camera.pos))
    forward = np.asarray(left_camera.forward, dtype=np.float64)
    forward /= np.linalg.norm(forward)
    up = np.asarray(left_camera.up, dtype=np.float64)
    up /= np.linalg.norm(up)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    relative = np.asarray(point_world, dtype=np.float64) - position
    depth = float(relative @ forward)
    if depth <= 1.0e-6 or bool(left_camera.orthographic):
        raise RuntimeError("third-person X500 projection is behind or incompatible with the camera")
    tangent_y = float(left_camera.frustum_top / left_camera.frustum_near)
    tangent_x = tangent_y * float(frame_width) / float(frame_height)
    x = 0.5 * frame_width * (1.0 + float(relative @ right) / (depth * tangent_x))
    y = 0.5 * frame_height * (1.0 - float(relative @ up) / (depth * tangent_y))
    return x, y, depth


def draw_drone_locator(
    frame: np.ndarray,
    *,
    center: tuple[float, float],
    box_size: tuple[float, float],
) -> np.ndarray:
    """Add a thin, projection-tracked locator without covering the X500."""
    center_x, center_y = center
    box_width, box_height = box_size
    margin_x = int(round(frame.shape[1] * 0.03))
    margin_y = int(round(frame.shape[0] * 0.03))
    # The exact segmentation samples already require the *whole* silhouette
    # to clear this margin. Between samples the perspective-scaled stale box
    # can temporarily overestimate the new silhouette, so require the current
    # projected vehicle centre (not the annotation box) to remain in view.
    if not (
        margin_x <= center_x < frame.shape[1] - margin_x
        and margin_y <= center_y < frame.shape[0] - margin_y
    ):
        raise RuntimeError("projected third-person X500 centre leaves the viewport safety margin")
    x0 = max(0, int(np.floor(center_x - 0.5 * box_width)) - 7)
    x1 = min(frame.shape[1] - 1, int(np.ceil(center_x + 0.5 * box_width)) + 7)
    y0 = max(0, int(np.floor(center_y - 0.5 * box_height)) - 7)
    y1 = min(frame.shape[0] - 1, int(np.ceil(center_y + 0.5 * box_height)) + 7)
    image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    colour = (55, 235, 255)
    cv2.rectangle(image, (x0, y0), (x1, y1), colour, 2, cv2.LINE_AA)
    label_y = y0 - 8 if y0 >= 28 else min(frame.shape[0] - 8, y1 + 21)
    cv2.putText(image, "X500", (x0, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (3, 7, 12), 3, cv2.LINE_AA)
    cv2.putText(image, "X500", (x0, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, colour, 1, cv2.LINE_AA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def animate_propellers(environment: Go2BackQrLandingEnv, propeller_ids: np.ndarray, base_quaternions: np.ndarray) -> None:
    """Render-only counter-rotating X500 props; all policy physics is unchanged."""
    angle = 2.0 * np.pi * 11.3 * float(environment.data.time)
    for index, (geom_id, base) in enumerate(zip(propeller_ids, base_quaternions)):
        half = (angle if index < 2 else -angle) / 2.0
        spin = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
        w0, x0, y0, z0 = spin
        w1, x1, y1, z1 = base
        environment.model.geom_quat[geom_id] = (
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        )
    # Propellers are visual-only.  Update their world rotation matrices
    # directly instead of calling any MuJoCo forward/kinematics routine during
    # inference: that keeps recorded state evolution bit-for-bit independent
    # of the renderer and prevents capture-only landing divergence.
    for geom_id in propeller_ids:
        local_rotation = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(local_rotation, environment.model.geom_quat[geom_id])
        body_id = int(environment.model.geom_bodyid[geom_id])
        environment.data.geom_xmat[geom_id] = (
            environment.data.xmat[body_id].reshape(3, 3) @ local_rotation.reshape(3, 3)
        ).reshape(9)


def draw_third_person_hud(
    frame: np.ndarray,
    *,
    time_s: float,
    error: float,
    altitude: float,
    path_distance: float,
    contacts: int,
    go2_speed: float,
    go2_slip: float,
    go2_tilt: float,
    detected: bool,
    imu_impact: bool,
    retry_active: bool,
    retry_count: int,
    terrain_label: str,
) -> np.ndarray:
    image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    lines = (
        "MuJoCo + ONNX Runtime | Unitree Go2 moving QR deck",
        f"t={time_s:05.1f}s  QR error={error:.3f} m  relative altitude={altitude:.2f} m",
        f"OFFLINE PHYSICS (NOT A SENSOR): path={path_distance:.2f} m | skid rails={contacts}/2",
        f"GO2 speed={go2_speed:.2f} m/s  | stance slip={go2_slip:.2f} m/s  | tilt={go2_tilt:.1f} deg",
        f"TERRAIN: {terrain_label}",
        f"CAMERA POLICY: {'QR DETECTED / TRACK' if detected else 'NO QR / CORRIDOR SEARCH'}",
        f"ONBOARD LANDING: IMU settle={'ON' if imu_impact else 'OFF'} | retry={'ON' if retry_active else 'OFF'} ({retry_count})",
    )
    for index, line in enumerate(lines):
        position = (18, 28 + 22 * index)
        # A filled HUD panel could occlude the X500 while the follow camera
        # reframes a tall approach.  Outlined glyphs keep the entire rendered
        # scene visible and remain readable on both sky and ground.
        cv2.putText(image, line, position, cv2.FONT_HERSHEY_SIMPLEX, 0.48, (3, 7, 12), 3, cv2.LINE_AA)
        cv2.putText(image, line, position, cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 247, 255), 1, cv2.LINE_AA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def draw_down_hud(
    frame: np.ndarray,
    *,
    error: float,
    contacts: int,
    force: float,
    penetration: float,
    detected: bool,
    imu_impact: bool,
    retry_active: bool,
    retry_count: int,
) -> np.ndarray:
    image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    height, width = image.shape[:2]
    lines = (
        f"ATTACHED DOWN CAMERA | {'QR DETECTED' if detected else 'SEARCHING QR'}",
        f"QR error={error:.3f} m  | offline sim skid rails={contacts}/2",
        f"offline sim normal={force:.2f} N  | penetration={penetration * 1000:.3f} mm",
        f"onboard IMU settle={'ON' if imu_impact else 'OFF'} | retry={'ON' if retry_active else 'OFF'} ({retry_count})",
    )
    for index, line in enumerate(lines):
        position = (18, 30 + 22 * index)
        # Use a thin text outline instead of the old opaque 110 px black HUD
        # panel.  The downward-camera pixels remain visible everywhere except
        # for the glyphs themselves.
        cv2.putText(image, line, position, cv2.FONT_HERSHEY_SIMPLEX, 0.43, (3, 7, 12), 3, cv2.LINE_AA)
        cv2.putText(image, line, position, cv2.FONT_HERSHEY_SIMPLEX, 0.43, (235, 247, 255), 1, cv2.LINE_AA)
    cv2.drawMarker(image, (width // 2, height // 2), (55, 235, 255), cv2.MARKER_CROSS, 28, 2, cv2.LINE_AA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def compose_dual_view(third_person: np.ndarray, down_camera: np.ndarray) -> np.ndarray:
    if third_person.shape != (720, 1280, 3):
        raise ValueError(f"unexpected third-person frame shape: {third_person.shape}")
    if down_camera.shape != (720, 640, 3):
        raise ValueError(f"unexpected downward-camera frame shape: {down_camera.shape}")
    frame = np.zeros((720, 1920, 3), dtype=np.uint8)
    frame[:, :1280] = third_person
    # Fill the complete right-hand panel.  The previous 640x540 inset left
    # 90 px black bars above and below the camera and made the view appear
    # partially blocked.
    frame[:, 1280:1920] = down_camera
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-model", type=Path, required=True)
    parser.add_argument(
        "--locomotion-model",
        type=Path,
        default=None,
        help="Optional MuJoCo PPO trained from legged-loco's Go2 task contract.",
    )
    parser.add_argument("--difficulty", choices=("easy", "medium", "hard"), default="medium")
    parser.add_argument("--terrain-task", choices=TERRAIN_TASKS, default="flat")
    parser.add_argument("--rough-level", type=int, choices=(1, 2, 3), default=None)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--video-file", type=Path, required=True)
    parser.add_argument("--snapshot-file", type=Path, required=True)
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional cap; by default use the selected difficulty's full environment horizon.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.onnx_model.exists():
        raise SystemExit(f"Missing ONNX model: {args.onnx_model}")
    session = ort.InferenceSession(str(args.onnx_model), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    environment = Go2BackQrLandingEnv(
        seed=args.seed,
        difficulty=args.difficulty,
        locomotion_model=args.locomotion_model,
        terrain_task=args.terrain_task,
        rough_level=args.rough_level,
    )
    observation, _ = environment.reset(seed=args.seed)
    propeller_ids = np.array(
        [mujoco.mj_name2id(environment.model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in ("propeller_front_right", "propeller_rear_left", "propeller_front_left", "propeller_rear_right")],
        dtype=np.int32,
    )
    propeller_base_quaternions = environment.model.geom_quat[propeller_ids].copy()
    args.video_file.parent.mkdir(parents=True, exist_ok=True)
    args.snapshot_file.parent.mkdir(parents=True, exist_ok=True)
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(environment.model, height=720, width=1280)
    drone_geom_ids = drone_render_geom_ids(environment)
    down_camera_option = mujoco.MjvOption()
    down_camera_option.geomgroup[1] = 0  # no X500 airframe/prop occlusion in its own camera view
    landed = False
    last_frame: np.ndarray | None = None
    last_render_time: float | None = None
    next_frame_time = 0.0
    last_drone_segmentation: np.ndarray | None = None
    last_drone_pixels: int | None = None
    last_drone_box_size: tuple[float, float] | None = None
    last_drone_projection_depth: float | None = None
    rendered_frame_index = 0
    with args.log_file.open("w", newline="", encoding="utf-8") as log_handle, imageio.get_writer(
        args.video_file, fps=args.fps, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=1
    ) as writer:
        csv_writer = csv.writer(log_handle)
        csv_writer.writerow([
            "sim_time_s", "qr_error_m", "altitude_m", "offline_sim_landing_skid_contacts",
            "offline_sim_landing_normal_force_n", "offline_sim_max_contact_penetration_m",
            "offline_sim_visual_contact_plane_error_m",
            "offline_sim_go2_path_distance_m",
            "offline_sim_pad_speed_mps", "offline_sim_go2_speed_mps", "offline_sim_go2_stance_slip_mps",
            "offline_sim_go2_base_height_m", "offline_sim_go2_tilt_deg", "offline_sim_go2_root_wrench_max_abs",
            "offline_sim_terrain_ground_height_m", "offline_sim_terrain_rough_level",
            "detected", "qr_center_u", "qr_center_v", "qr_pnp_depth_m",
            "qr_center_rate_u", "qr_center_rate_v", "imu_impact_latched",
            "landing_retry_active", "landing_retry_count", "onnx_provider",
            "third_person_drone_pixels", "down_view_nonblack_fraction",
            "down_view_luma_std", "dual_view_state_match",
            "third_person_visibility_sampled",
            "third_person_projection_visible",
        ])

        def render_substep(current: Go2BackQrLandingEnv, *, force: bool = False) -> None:
            nonlocal last_frame, last_render_time, next_frame_time
            nonlocal last_drone_segmentation, last_drone_pixels, rendered_frame_index
            nonlocal last_drone_box_size, last_drone_projection_depth
            current_time = float(current.data.time)
            if not force and current_time + 1.0e-9 < next_frame_time:
                return
            # A terminal 5 ms substep may coincide exactly with an ordinary
            # 30 Hz capture.  Do not duplicate it when that state was already
            # written, but force one final frame when cadence skipped it.
            if force and last_render_time is not None and abs(current_time - last_render_time) <= 1.0e-9:
                return
            animate_propellers(current, propeller_ids, propeller_base_quaternions)
            error = float(np.linalg.norm(current._horizontal_error()))
            altitude = float(current._relative_altitude())
            # All values below that depend on MuJoCo body/contact truth are
            # offline_sim scoring/HUD diagnostics, never X500 policy inputs.
            contacts = int(current._offline_sim_landing_skid_contact_count)
            detected = bool(current._qr_detected)
            imu_impact = bool(current._imu_impact_latched)
            retry_active = bool(current._landing_retry_active)
            retry_count = int(current._landing_retry_count)
            go2_speed = float(np.linalg.norm(current.data.qvel[:2]))
            go2_tilt = float(np.degrees(np.arccos(np.clip(current.data.xmat[current.base_id, 8], -1.0, 1.0))))
            synchronized_state = render_state_fingerprint(current)
            third_person_camera = follow_camera(current)
            renderer.update_scene(current.data, camera=third_person_camera)
            third_rgb = renderer.render().copy()
            projected_x, projected_y, projection_depth = project_world_point(
                renderer,
                current.drone_position,
                frame_width=third_rgb.shape[1],
                frame_height=third_rgb.shape[0],
            )
            assert_render_state_unchanged(
                current, synchronized_state, rendered_view="third-person RGB"
            )
            visibility_sampled = (
                last_drone_segmentation is None
                or rendered_frame_index % SEGMENTATION_SAMPLE_INTERVAL_FRAMES == 0
                or force
            )
            if visibility_sampled:
                renderer.enable_segmentation_rendering()
                try:
                    segmentation = renderer.render().copy()
                finally:
                    renderer.disable_segmentation_rendering()
                third_person_drone_pixels = assert_drone_visible_in_segmentation(
                    segmentation, drone_geom_ids=drone_geom_ids
                )
                assert_render_state_unchanged(
                    current, synchronized_state, rendered_view="third-person segmentation"
                )
                last_drone_segmentation = segmentation
                last_drone_pixels = third_person_drone_pixels
                _, _, box_width, box_height = drone_segmentation_box(
                    segmentation, drone_geom_ids=drone_geom_ids
                )
                last_drone_box_size = (box_width, box_height)
                last_drone_projection_depth = projection_depth
            else:
                # The periodic segmentation proves the actual rendered
                # silhouette.  Between samples, MuJoCo's current GL-camera
                # projection moves and scales the locator every frame.
                assert last_drone_segmentation is not None and last_drone_pixels is not None
                segmentation = last_drone_segmentation
                third_person_drone_pixels = last_drone_pixels
            assert last_drone_box_size is not None and last_drone_projection_depth is not None
            projection_scale = last_drone_projection_depth / projection_depth
            third = draw_drone_locator(
                third_rgb,
                center=(projected_x, projected_y),
                box_size=(
                    last_drone_box_size[0] * projection_scale,
                    last_drone_box_size[1] * projection_scale,
                ),
            )
            third = draw_third_person_hud(
                third, time_s=float(current.data.time), error=error, altitude=altitude,
                path_distance=float(current._path_length), contacts=contacts, go2_speed=go2_speed,
                go2_slip=float(current._go2_stance_slip_mps), go2_tilt=go2_tilt, detected=detected,
                imu_impact=imu_impact, retry_active=retry_active, retry_count=retry_count,
                terrain_label=terrain_hud_label(
                    current.terrain_task,
                    current._active_rough_level if current.terrain_task == "rough" else None,
                ),
            )
            renderer.update_scene(current.data, camera="down_camera", scene_option=down_camera_option)
            down_wide = renderer.render().copy()
            assert_render_state_unchanged(
                current, synchronized_state, rendered_view="down-camera RGB"
            )
            # The common renderer is 1280x720.  A centered 640x720 crop keeps
            # the camera geometry undistorted and fills the complete portrait
            # side panel without black letterboxing.
            down = down_wide[:, 320:960].copy()
            down_view_nonblack_fraction = float(np.mean(np.any(down > 8, axis=2)))
            down_view_luma_std = float(cv2.cvtColor(down, cv2.COLOR_RGB2GRAY).std())
            down = draw_down_hud(
                down, error=error, contacts=contacts,
                force=float(current._offline_sim_landing_normal_force),
                penetration=float(current._offline_sim_max_contact_penetration), detected=detected,
                imu_impact=imu_impact, retry_active=retry_active, retry_count=retry_count,
            )
            last_frame = compose_dual_view(third, down)
            writer.append_data(last_frame)
            csv_writer.writerow([
                f"{current.data.time:.3f}", f"{error:.6f}", f"{altitude:.6f}", contacts,
                f"{current._offline_sim_landing_normal_force:.6f}",
                f"{current._offline_sim_max_contact_penetration:.8f}",
                f"{abs(X500_SKID_CENTER_BODY_Z_M - X500_SKID_HALF_SIZE_M[2] - X500_VISUAL_SKID_BOTTOM_BODY_Z_M):.8f}",
                f"{current._path_length:.6f}",
                f"{np.linalg.norm(current._pad_velocity):.6f}", f"{go2_speed:.6f}",
                f"{current._go2_stance_slip_mps:.6f}", f"{current.base_position[2]:.6f}", f"{go2_tilt:.6f}",
                f"{np.max(np.abs(current.data.xfrc_applied[current.base_id])):.6f}",
                f"{terrain_height_at(current.terrain_task, float(current.base_position[0]), float(current.base_position[1]), rough_level=current._active_rough_level if current.terrain_task == 'rough' else None):.6f}",
                int(current._active_rough_level) if current.terrain_task == "rough" else 0,
                int(detected), f"{current._qr_center_norm[0]:.6f}", f"{current._qr_center_norm[1]:.6f}",
                f"{current._qr_depth if detected else 0.0:.6f}", f"{current._qr_center_rate[0]:.6f}",
                f"{current._qr_center_rate[1]:.6f}", int(imu_impact), int(retry_active),
                retry_count, "CPUExecutionProvider", third_person_drone_pixels,
                f"{down_view_nonblack_fraction:.6f}", f"{down_view_luma_std:.6f}", 1,
                int(visibility_sampled), 1,
            ])
            last_render_time = current_time
            rendered_frame_index += 1
            if not force:
                next_frame_time += 1.0 / float(args.fps)

        environment.physics_observer = render_substep
        step_limit = environment.max_steps if args.max_steps is None else min(args.max_steps, environment.max_steps)
        if step_limit <= 0:
            raise ValueError("--max-steps must be positive")
        for _ in range(step_limit):
            action = session.run([output_name], {input_name: observation[np.newaxis, :].astype(np.float32)})[0][0]
            action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
            observation, _, terminated, truncated, info = environment.step(action)
            if terminated or truncated:
                render_substep(environment, force=True)
                landed = bool(info["success"] > 0.5)
                break
        environment.physics_observer = None
    renderer.close()
    if last_frame is None:
        raise RuntimeError("Go2 ONNX inference did not render a frame")
    imageio.imwrite(args.snapshot_file, last_frame)
    environment.close()
    if not landed:
        raise SystemExit("Go2 ONNX inference finished without a stable QR-deck landing")
    print(f"GO2 LAND confirmed: {args.video_file} | snapshot: {args.snapshot_file}")


if __name__ == "__main__":
    main()
