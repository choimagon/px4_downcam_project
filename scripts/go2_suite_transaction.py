#!/usr/bin/env python3
"""Validate, finalize, and transactionally publish the Go2 QR video suite."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.util
import json
import math
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALGORITHMS = ("ppo", "ddpg", "sac")
DIFFICULTIES = ("easy", "medium", "hard")
MAX_OFFLINE_SIM_PENETRATION_M = 0.002
MAX_VISUAL_CONTACT_PLANE_ERROR_M = 0.001
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 720
DOWN_VIEW_X_START = 1280
DOWN_VIEW_SAMPLE_COUNT = 7
# Analyse only the central/lower camera image. This excludes the legacy top
# HUD and letterbox edges, so text or borders cannot make a black feed pass.
DOWN_VIEW_ROI_Y_FRACTION = (0.30, 0.85)
DOWN_VIEW_ROI_X_FRACTION = (0.05, 0.95)
DOWN_VIEW_BLACK_LUMA = 12
DOWN_VIEW_MIN_NONBLACK_FRACTION = 0.10
DOWN_VIEW_MIN_LUMA_STD = 3.0
DOWN_VIEW_MIN_EDGE_ENERGY = 0.05
DOWN_VIEW_MIN_MEDIAN_TEMPORAL_MAD = 0.50
DOWN_VIEW_MIN_MAX_TEMPORAL_MAD = 1.00
MIN_THIRD_PERSON_DRONE_PIXELS = 320.0
SEGMENTATION_SAMPLE_INTERVAL_FRAMES = 30
MIN_RUNTIME_DOWN_VIEW_NONBLACK_FRACTION = 0.25
MIN_RUNTIME_DOWN_VIEW_LUMA_STD = 2.0
OBSERVATION_NAMES = (
    "qr_center_u",
    "qr_center_v",
    "qr_pnp_depth",
    "qr_detected",
    "qr_center_rate_u",
    "qr_center_rate_v",
    "drone_vertical_velocity",
)
# These semantic markers deliberately live in the generated HTML rather than
# being inferred from incidental prose.  They make publication fail if a
# future dashboard rewrite drops the practical sensor/acquisition guide while
# leaving only the equations behind.
DASHBOARD_INPUT_GUIDE_MARKERS = (
    ("detailed 7D input guide", 'id="drone-input-guide"'),
    ("camera-to-vector pipeline", 'id="camera-to-vector-pipeline"'),
    ("PX4 NED source frame", 'data-source-frame="PX4-NED"'),
    ("PX4 vertical-velocity sign conversion", 'data-vz-conversion="negate"'),
    ("policy-camera resolution", 'data-policy-camera-resolution="1280x960"'),
    ("video down-view resolution", 'data-video-down-view-resolution="640x720"'),
    ("real-hardware adapter implementation status", 'data-real-adapter-status="not-implemented"'),
)
DASHBOARD_INPUT_GUIDE_VISIBLE_TEXT = (
    "드론 정책 입력 7개",
    "모델 필드",
    "무슨 값인가",
    "원시값을 어떻게 얻나",
    "모델 입력으로 가공",
    "갱신·유실 규칙",
    "실기 출처와 현재 상태",
    "카메라 프레임",
    "QR 코너 검출",
    "PnP",
    "PX4 NED",
    "부호",
    "정책 카메라",
    "1280×960",
    "640×720",
    "시각화용",
    "실기 어댑터",
    "구현되지 않았",
)
PRIMARY_METRICS = (
    "mean_reward",
    "std_reward",
    "success_rate",
    "mean_terminal_error_m",
    "mean_episode_duration_s",
    "mean_episode_steps",
)
INFERENCE_CSV_FIELDS = {
    "sim_time_s",
    "qr_error_m",
    "altitude_m",
    "detected",
    "qr_center_u",
    "qr_center_v",
    "qr_pnp_depth_m",
    "qr_center_rate_u",
    "qr_center_rate_v",
    "imu_impact_latched",
    "landing_retry_active",
    "landing_retry_count",
    "offline_sim_landing_skid_contacts",
    "offline_sim_landing_normal_force_n",
    "offline_sim_max_contact_penetration_m",
    "offline_sim_visual_contact_plane_error_m",
    "offline_sim_go2_path_distance_m",
    "offline_sim_pad_speed_mps",
    "offline_sim_go2_speed_mps",
    "offline_sim_go2_stance_slip_mps",
    "offline_sim_go2_base_height_m",
    "offline_sim_go2_tilt_deg",
    "offline_sim_go2_root_wrench_max_abs",
    "third_person_drone_pixels",
    "down_view_nonblack_fraction",
    "down_view_luma_std",
    "dual_view_state_match",
    "third_person_visibility_sampled",
    "third_person_projection_visible",
    "onnx_provider",
}
LEGACY_CSV_FIELDS = {
    "gear_contacts", "offline_sim_landing_foot_contacts",
    "pad_speed_mps", "go2_assist_force_n",
}
VIDEO_SEEDS = {
    ("ppo", "easy"): 20261121,
    ("ppo", "medium"): 20261021,
    ("ppo", "hard"): 20261219,
    ("ddpg", "easy"): 20261123,
    ("ddpg", "medium"): 20261023,
    ("ddpg", "hard"): 20261218,
    ("sac", "easy"): 20261127,
    ("sac", "medium"): 20261027,
    ("sac", "hard"): 20261203,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def evenly_spaced_frame_indices(frame_count: int, sample_count: int = DOWN_VIEW_SAMPLE_COUNT) -> list[int]:
    """Return deterministic start-to-end sample indices, including both ends."""
    if frame_count <= 0 or sample_count <= 0:
        return []
    if frame_count == 1:
        return [0]
    count = min(frame_count, sample_count)
    return sorted(
        {
            int(round(index * (frame_count - 1) / (count - 1)))
            for index in range(count)
        }
    )


def down_view_quality_metrics(frame: Any) -> tuple[dict[str, float], Any]:
    """Measure the HUD-independent part of the 640x720 right camera panel.

    The low-resolution signature is used only for temporal freeze detection.
    X500 identity is deliberately not guessed from RGB colours or HUD pixels;
    the inference renderer supplies segmentation-derived pixel evidence.
    """
    import cv2
    import numpy as np

    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] < 3:
        raise ValueError("decoded video frame is not an HxWx3 image")
    if frame.shape[:2] != (VIDEO_HEIGHT, VIDEO_WIDTH):
        raise ValueError(
            f"decoded video frame is {frame.shape[1]}x{frame.shape[0]}, "
            f"expected {VIDEO_WIDTH}x{VIDEO_HEIGHT}"
        )
    right = frame[:, DOWN_VIEW_X_START:VIDEO_WIDTH]
    height, width = right.shape[:2]
    y0 = int(round(height * DOWN_VIEW_ROI_Y_FRACTION[0]))
    y1 = int(round(height * DOWN_VIEW_ROI_Y_FRACTION[1]))
    x0 = int(round(width * DOWN_VIEW_ROI_X_FRACTION[0]))
    x1 = int(round(width * DOWN_VIEW_ROI_X_FRACTION[1]))
    roi = right[y0:y1, x0:x1]
    if roi.size == 0:
        raise ValueError("down-view quality ROI is empty")
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    signature = cv2.resize(gray, (96, 64), interpolation=cv2.INTER_AREA)
    signature = cv2.GaussianBlur(signature, (5, 5), 0).astype(np.float32)
    return (
        {
            "nonblack_fraction": float(np.mean(gray > DOWN_VIEW_BLACK_LUMA)),
            "luma_std": float(np.std(gray)),
            "edge_energy": float(np.mean(np.abs(laplacian))),
        },
        signature,
    )


def validate_down_view_video(capture: Any, frame_count: int) -> list[str]:
    """Decode uniform samples and reject black, blank, or frozen down views."""
    import cv2
    import numpy as np

    issues: list[str] = []
    indices = evenly_spaced_frame_indices(frame_count)
    if len(indices) < 3:
        return [f"MP4 has only {frame_count} frame(s); down-view quality needs at least 3"]

    signatures: list[Any] = []
    decoded_indices: list[int] = []
    for frame_index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok or frame is None:
            issues.append(f"down-view sample frame {frame_index} is not decodable")
            continue
        try:
            metrics, signature = down_view_quality_metrics(frame)
        except ValueError as error:
            issues.append(f"down-view sample frame {frame_index}: {error}")
            continue
        decoded_indices.append(frame_index)
        signatures.append(signature)
        if metrics["nonblack_fraction"] < DOWN_VIEW_MIN_NONBLACK_FRACTION:
            issues.append(
                f"down-view sample frame {frame_index} is black "
                f"(nonblack={metrics['nonblack_fraction']:.3f} < "
                f"{DOWN_VIEW_MIN_NONBLACK_FRACTION:.3f})"
            )
        if metrics["luma_std"] < DOWN_VIEW_MIN_LUMA_STD:
            issues.append(
                f"down-view sample frame {frame_index} is visually blank "
                f"(luma std={metrics['luma_std']:.3f} < {DOWN_VIEW_MIN_LUMA_STD:.3f})"
            )
        if metrics["edge_energy"] < DOWN_VIEW_MIN_EDGE_ENERGY:
            issues.append(
                f"down-view sample frame {frame_index} lacks scene structure "
                f"(edge energy={metrics['edge_energy']:.3f} < "
                f"{DOWN_VIEW_MIN_EDGE_ENERGY:.3f})"
            )

    if len(signatures) >= 2:
        changes = [
            float(np.mean(np.abs(current - previous)))
            for previous, current in zip(signatures, signatures[1:])
        ]
        median_change = float(np.median(changes))
        max_change = float(np.max(changes))
        if (
            median_change < DOWN_VIEW_MIN_MEDIAN_TEMPORAL_MAD
            or max_change < DOWN_VIEW_MIN_MAX_TEMPORAL_MAD
        ):
            issues.append(
                "down-view appears frozen across sampled frames "
                f"{decoded_indices} (median MAD={median_change:.3f}, "
                f"max MAD={max_change:.3f})"
            )
    else:
        issues.append("fewer than 2 down-view samples decoded; temporal quality is unknown")
    return issues


def resolved_from_project(project_root: Path, value: object) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    return (path if path.is_absolute() else project_root / path).resolve()


def preflight(project_root: Path, artifacts: Path, models: Path, locomotion_model: Path) -> dict[str, Any]:
    """Reject stale, privileged, partially promoted, or unaccepted model sets."""
    errors: list[str] = []
    landing_metrics_path = artifacts / "go2_back_qr_training_metrics.json"
    locomotion_metrics_path = artifacts / "go2_legged_loco_metrics.json"
    for path in (landing_metrics_path, locomotion_metrics_path, locomotion_model):
        if not path.is_file():
            errors.append(f"missing required file: {path}")
    if errors:
        raise RuntimeError("preflight failed:\n- " + "\n- ".join(errors))

    landing = read_json(landing_metrics_path)
    locomotion = read_json(locomotion_metrics_path)
    locomotion_acceptance = locomotion.get("acceptance", {})
    if not isinstance(locomotion_acceptance, dict) or locomotion_acceptance.get("passed") is not True:
        errors.append("low-level Go2 locomotion acceptance.passed is not true")
        locomotion_acceptance = {}
    failed_locomotion_criteria = sorted(
        key
        for key, value in locomotion_acceptance.items()
        if key != "passed" and value is not True
    )
    if failed_locomotion_criteria:
        errors.append(
            "low-level Go2 locomotion has failed acceptance criteria: "
            + ", ".join(failed_locomotion_criteria)
        )
    declared_locomotion = resolved_from_project(project_root, locomotion.get("model"))
    if declared_locomotion != locomotion_model.resolve():
        errors.append(
            "low-level metrics model does not identify the selected canonical model: "
            f"{declared_locomotion} != {locomotion_model.resolve()}"
        )
    if locomotion_metrics_path.stat().st_mtime_ns < locomotion_model.stat().st_mtime_ns:
        errors.append("low-level locomotion metrics predates the selected model bytes")

    if landing.get("algorithms") != list(ALGORITHMS):
        errors.append(f"landing algorithms must be exactly {list(ALGORITHMS)}")
    if landing.get("training_difficulty") != "train":
        errors.append("landing training_difficulty must be train")
    if landing.get("evaluation_difficulties") != list(DIFFICULTIES):
        errors.append(f"landing evaluation_difficulties must be exactly {list(DIFFICULTIES)}")
    if tuple(landing.get("observation_names", ())) != OBSERVATION_NAMES:
        errors.append("landing observation_names is not the camera/PX4 sensor-only 7D contract")
    if landing.get("primary_evaluation_metrics") != list(PRIMARY_METRICS):
        errors.append("landing primary_evaluation_metrics is not the required six-field contract")
    if landing.get("sensor_rates_hz") != {"downward_camera": 30, "px4_estimator": 50}:
        errors.append("landing sensor_rates_hz must be camera=30 and PX4 estimator=50")
    if landing.get("training_policy_residual_speed_mps") != 0.002:
        errors.append("landing training exploration residual must be exactly 0.002 m/s")
    if landing.get("deployment_policy_residual_speed_mps") != 0.001:
        errors.append("landing held-out/deployment residual must be exactly 0.001 m/s")
    if landing.get("max_offline_sim_penetration_m") != MAX_OFFLINE_SIM_PENETRATION_M:
        errors.append("landing max offline simulator penetration must be exactly 0.002 m")
    if "offline_sim_worst_max_penetration_m" not in landing.get("offline_sim_diagnostics", ()):
        errors.append("landing metrics omit offline_sim_worst_max_penetration_m")
    geometry = landing.get("x500_landing_geometry", {})
    if not isinstance(geometry, dict):
        errors.append("landing metrics omit the X500 visual/contact geometry contract")
        geometry = {}
    visual_sole = geometry.get("visual_skid_sole_body_z_m")
    contact_sole = geometry.get("rail_sole_body_z_m")
    declared_plane_error = geometry.get("visual_contact_plane_error_m")
    if not all(finite_number(value) for value in (visual_sole, contact_sole, declared_plane_error)):
        errors.append("landing X500 visual/contact sole calibration is non-numeric or missing")
    elif (
        abs(float(visual_sole) - float(contact_sole)) > MAX_VISUAL_CONTACT_PLANE_ERROR_M
        or float(declared_plane_error) > MAX_VISUAL_CONTACT_PLANE_ERROR_M
    ):
        errors.append("landing X500 rendered skid and contact sole differ by more than 1 mm")
    if geometry.get("rail_center_xy_m") != [[0.0, 0.132], [0.0, -0.132]]:
        errors.append("landing X500 rails are not at the stock Gazebo lateral positions")
    if geometry.get("rail_half_size_m") != [0.125, 0.0075, 0.0075]:
        errors.append("landing X500 rail dimensions do not match the stock Gazebo model")
    if geometry.get("rail_condim") != 3:
        errors.append("landing X500 rail contact dimension must be 3")
    if geometry.get("rails_rendered") is not True:
        errors.append("landing X500 landing-sole collision objects must be visible in recordings")
    if geometry.get("camera_placeholder_rendered") is not False:
        errors.append("landing decorative camera box must remain hidden in recordings")
    camera_depth = geometry.get("touchdown_down_camera_depth_m")
    if not finite_number(camera_depth) or float(camera_depth) <= 0.10:
        errors.append("landing down camera must remain outside its 0.10 m near clip at skid touchdown")
    # Validate the current source assets and compiled MJCF too, rather than
    # trusting metadata written by the trainer.  This catches a future mesh,
    # offset, alpha, or contact-position change before any stale-looking MP4
    # can replace the currently published suite.
    geometry_env = None
    try:
        from landing_rl.go2_qr_environment import Go2BackQrLandingEnv

        frame_mesh = project_root / "assets" / "mujoco_x500" / "x500_frame.obj"
        raw_vertex_z = [
            float(line.split()[3])
            for line in frame_mesh.read_text(encoding="utf-8").splitlines()
            if line.startswith("v ")
        ]
        if not raw_vertex_z:
            raise RuntimeError(f"X500 frame mesh has no vertices: {frame_mesh}")
        current_visual_sole = min(raw_vertex_z) + 0.025
        geometry_env = Go2BackQrLandingEnv(seed=0, difficulty="easy")
        expected_xy = ((0.0, 0.132), (0.0, -0.132))
        for skid_id, xy in zip(geometry_env.skid_ids, expected_xy):
            skid_id = int(skid_id)
            current_contact_sole = float(
                geometry_env.model.geom_pos[skid_id, 2]
                - geometry_env.model.geom_size[skid_id, 2]
            )
            if abs(current_contact_sole - current_visual_sole) > MAX_VISUAL_CONTACT_PLANE_ERROR_M:
                errors.append("compiled X500 rendered skid/contact sole mismatch exceeds 1 mm")
            if any(
                abs(float(actual) - expected) > 1.0e-9
                for actual, expected in zip(geometry_env.model.geom_pos[skid_id, :2], xy)
            ):
                errors.append("compiled X500 rail is not at its stock Gazebo position")
            if any(
                abs(float(actual) - expected) > 1.0e-9
                for actual, expected in zip(
                    geometry_env.model.geom_size[skid_id], (0.125, 0.0075, 0.0075)
                )
            ):
                errors.append("compiled X500 rail dimensions differ from stock Gazebo")
            if int(geometry_env.model.geom_condim[skid_id]) != 3:
                errors.append("compiled X500 rail contact dimension is not 3")
            if float(geometry_env.model.geom_rgba[skid_id, 3]) != 1.0:
                errors.append("compiled X500 landing-sole collision object is not visible")
        landing_surface = geometry_env.model.geom("landing_surface")
        qr_print = geometry_env.model.geom("qr_black_nw")
        ink_clearance = float(
            qr_print.pos[2] + qr_print.size[2]
            - landing_surface.pos[2] - landing_surface.size[2]
        )
        if not 0.0 < ink_clearance <= 0.000004:
            errors.append("compiled QR ink render clearance must be positive and at most 4 micrometres")
        rotor_pairs = (
            ("propeller_front_right", "rotor_axis_front_right"),
            ("propeller_rear_left", "rotor_axis_rear_left"),
            ("propeller_front_left", "rotor_axis_front_left"),
            ("propeller_rear_right", "rotor_axis_rear_right"),
        )
        import numpy as np
        import mujoco
        mujoco.mj_forward(geometry_env.model, geometry_env.data)
        for propeller_name, axis_name in rotor_pairs:
            propeller = geometry_env.model.geom(propeller_name)
            axis = geometry_env.model.site(axis_name)
            if float(np.linalg.norm(
                geometry_env.data.geom_xpos[propeller.id, :2]
                - geometry_env.data.site_xpos[axis.id, :2]
            )) > 0.0001:
                errors.append(f"compiled {propeller_name} is not centered over {axis_name}")
        for name in ("mono_cam_housing", "mono_cam_lens"):
            if float(geometry_env.model.geom(name).rgba[3]) != 0.0:
                errors.append(f"compiled decorative X500 {name} is visible")
    except Exception as error:
        errors.append(f"cannot validate compiled X500 landing geometry: {error}")
    finally:
        if geometry_env is not None:
            geometry_env.close()

    promotion = landing.get("promotion", {})
    if not isinstance(promotion, dict) or promotion.get("passed") is not True:
        errors.append("landing three-model promotion.passed is not true")
        promotion = {}
    if promotion.get("status") not in {"promoted_atomically", "validated_existing"}:
        errors.append("landing promotion status is not an accepted atomic model set")
    if promotion.get("atomic_model_set") is not True:
        errors.append("landing promotion does not declare atomic_model_set=true")
    if promotion.get("required_algorithms") != list(ALGORITHMS):
        errors.append("landing promotion does not cover PPO, DDPG and SAC together")
    if promotion.get("required_profiles") != ["train", *DIFFICULTIES]:
        errors.append("landing promotion does not cover train/easy/medium/hard")

    landing_metrics = landing.get("metrics", {})
    if not isinstance(landing_metrics, dict) or set(landing_metrics) != set(ALGORITHMS):
        errors.append("landing metrics must contain exactly PPO, DDPG and SAC")
        landing_metrics = {}
    for algorithm in ALGORITHMS:
        algorithm_metrics = landing_metrics.get(algorithm, {})
        if algorithm_metrics.get("acceptance", {}).get("passed") is not True:
            errors.append(f"{algorithm} landing acceptance.passed is not true")
        held_out = algorithm_metrics.get("held_out", {})
        if not isinstance(held_out, dict) or set(held_out) != set(DIFFICULTIES):
            errors.append(f"{algorithm} held_out profiles must be exactly easy/medium/hard")
            held_out = {}
        profiles = {"train": algorithm_metrics.get("training", {})}
        profiles.update({difficulty: held_out.get(difficulty, {}) for difficulty in DIFFICULTIES})
        for profile_name, values in profiles.items():
            if not isinstance(values, dict):
                errors.append(f"{algorithm}/{profile_name} metrics is not an object")
                continue
            for field in PRIMARY_METRICS:
                if not finite_number(values.get(field)):
                    errors.append(f"{algorithm}/{profile_name} missing finite {field}")
            for field in (
                "offline_sim_go2_fall_rate",
                "offline_sim_go2_root_wrench_max_abs",
                "offline_sim_worst_max_penetration_m",
            ):
                if not finite_number(values.get(field)):
                    errors.append(f"{algorithm}/{profile_name} missing finite {field}")
            if finite_number(values.get("success_rate")) and float(values["success_rate"]) < 0.95:
                errors.append(f"{algorithm}/{profile_name} success rate is below 0.95")
            if finite_number(values.get("mean_terminal_error_m")) and float(values["mean_terminal_error_m"]) > 0.055:
                errors.append(f"{algorithm}/{profile_name} terminal error exceeds 0.055 m")
            if finite_number(values.get("offline_sim_go2_fall_rate")) and float(values["offline_sim_go2_fall_rate"]) != 0.0:
                errors.append(f"{algorithm}/{profile_name} Go2 fall rate is nonzero")
            if finite_number(values.get("offline_sim_go2_root_wrench_max_abs")) and float(values["offline_sim_go2_root_wrench_max_abs"]) != 0.0:
                errors.append(f"{algorithm}/{profile_name} root wrench is nonzero")
            if finite_number(values.get("offline_sim_worst_max_penetration_m")) and float(values["offline_sim_worst_max_penetration_m"]) > MAX_OFFLINE_SIM_PENETRATION_M:
                errors.append(f"{algorithm}/{profile_name} worst penetration exceeds 2 mm")

    canonical_hashes = promotion.get("canonical_model_sha256", {})
    if not isinstance(canonical_hashes, dict) or set(canonical_hashes) != set(ALGORITHMS):
        errors.append("landing promotion is missing the exact canonical SHA-256 set")
        canonical_hashes = {}
    canonical_paths = {algorithm: models / f"{algorithm}_go2_back_qr.zip" for algorithm in ALGORITHMS}
    for algorithm, path in canonical_paths.items():
        if not path.is_file():
            errors.append(f"missing canonical landing policy: {path}")
        elif canonical_hashes.get(algorithm) != sha256_file(path):
            errors.append(f"{algorithm} canonical SHA-256 does not match the accepted landing manifest")

    actual_locomotion_hash = sha256_file(locomotion_model)
    if landing.get("locomotion_model_sha256") != actual_locomotion_hash:
        errors.append("landing policies were not evaluated with this exact low-level locomotion model")

    if not errors:
        from stable_baselines3 import PPO
        from landing_rl.train import ALGORITHMS as SB3_ALGORITHMS

        try:
            low_level = PPO.load(locomotion_model, device="cpu")
            if tuple(low_level.observation_space.shape) != (450,):
                errors.append(f"low-level observation shape is {low_level.observation_space.shape}, expected (450,)")
            if tuple(low_level.action_space.shape) != (12,):
                errors.append(f"low-level action shape is {low_level.action_space.shape}, expected (12,)")
        except Exception as error:  # pragma: no cover - SB3 supplies backend-specific errors
            errors.append(f"cannot load low-level locomotion policy: {error}")
        for algorithm, path in canonical_paths.items():
            try:
                model = SB3_ALGORITHMS[algorithm].load(path, device="cpu")
                if tuple(model.observation_space.shape) != (len(OBSERVATION_NAMES),):
                    errors.append(
                        f"{algorithm} observation shape is {model.observation_space.shape}, "
                        f"expected ({len(OBSERVATION_NAMES)},)"
                    )
                if tuple(model.action_space.shape) != (2,):
                    errors.append(f"{algorithm} action shape is {model.action_space.shape}, expected (2,)")
            except Exception as error:  # pragma: no cover - SB3 supplies backend-specific errors
                errors.append(f"cannot load {algorithm} canonical landing policy: {error}")

    mathjax = artifacts / "vendor" / "node_modules" / "mathjax" / "es5" / "tex-mml-chtml.js"
    if not mathjax.is_file():
        errors.append(f"missing local MathJax runtime: {mathjax}")
    if errors:
        raise RuntimeError("preflight failed:\n- " + "\n- ".join(errors))
    return {
        "status": "passed",
        "algorithms": list(ALGORITHMS),
        "profiles": ["train", *DIFFICULTIES],
        "observation_names": list(OBSERVATION_NAMES),
        "landing_model_sha256": canonical_hashes,
        "locomotion_model_sha256": actual_locomotion_hash,
        "locomotion_model": str(locomotion_model.resolve()),
        "landing_metrics_sha256": sha256_file(landing_metrics_path),
        "locomotion_metrics_sha256": sha256_file(locomotion_metrics_path),
    }


def demo_stem(algorithm: str, difficulty: str) -> str:
    return f"{algorithm}_go2_back_qr_onnx_{difficulty}_follow"


def snapshot_inputs(
    project_root: Path, artifacts: Path, models: Path, locomotion_model: Path, generation: Path
) -> dict[str, Any]:
    """Copy one verified model/metrics generation for deterministic capture."""
    report = preflight(project_root, artifacts, models, locomotion_model)
    inputs = generation / "model_inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    for algorithm in ALGORITHMS:
        source = models / f"{algorithm}_go2_back_qr.zip"
        target = inputs / source.name
        shutil.copy2(source, target)
        if sha256_file(target) != report["landing_model_sha256"][algorithm]:
            raise RuntimeError(f"{algorithm} canonical changed while it was being snapshotted")
    locomotion_target = inputs / "go2_legged_loco_ppo.zip"
    shutil.copy2(locomotion_model, locomotion_target)
    if sha256_file(locomotion_target) != report["locomotion_model_sha256"]:
        raise RuntimeError("low-level locomotion model changed while it was being snapshotted")
    for filename, hash_key in (
        ("go2_back_qr_training_metrics.json", "landing_metrics_sha256"),
        ("go2_legged_loco_metrics.json", "locomotion_metrics_sha256"),
    ):
        source = artifacts / filename
        target = generation / filename
        shutil.copy2(source, target)
        if sha256_file(target) != report[hash_key]:
            raise RuntimeError(f"{filename} changed while it was being snapshotted")
    write_json_atomic(generation / "go2_back_qr_suite_preflight.json", report)
    return report


def normalize_onnx_manifest(generation: Path, artifacts: Path, models: Path) -> None:
    path = generation / "go2_back_qr_onnx_models.json"
    manifest = read_json(path)
    entries = manifest.get("models", [])
    by_algorithm = {
        entry.get("algorithm"): entry
        for entry in entries
        if isinstance(entry, dict) and entry.get("algorithm") in ALGORITHMS
    }
    if set(by_algorithm) != set(ALGORITHMS) or len(entries) != len(ALGORITHMS):
        raise RuntimeError("ONNX manifest must contain exactly PPO, DDPG and SAC")
    preflight_report = read_json(generation / "go2_back_qr_suite_preflight.json")
    normalized: list[dict[str, Any]] = []
    for algorithm in ALGORITHMS:
        entry = dict(by_algorithm[algorithm])
        declared_source = Path(str(entry.get("source", ""))).resolve()
        expected_source = (generation / "model_inputs" / f"{algorithm}_go2_back_qr.zip").resolve()
        if declared_source != expected_source or not declared_source.is_file():
            raise RuntimeError(f"{algorithm} ONNX exporter did not use the verified model snapshot")
        source_hash = sha256_file(declared_source)
        if source_hash != preflight_report.get("landing_model_sha256", {}).get(algorithm):
            raise RuntimeError(f"{algorithm} ONNX source hash does not match accepted preflight")
        onnx_path = generation / "onnx_go2" / f"{algorithm}_go2_back_qr.onnx"
        if not onnx_path.is_file():
            raise RuntimeError(f"missing staged {algorithm} ONNX: {onnx_path}")
        entry["source"] = str(models / f"{algorithm}_go2_back_qr.zip")
        entry["onnx"] = str(artifacts / "onnx_go2" / f"{algorithm}_go2_back_qr.onnx")
        entry["source_model_sha256"] = source_hash
        entry["onnx_sha256"] = sha256_file(onnx_path)
        normalized.append(entry)
    write_json_atomic(path, {"models": normalized})


def build_dashboard(project_root: Path, generation: Path, artifacts: Path, models: Path) -> None:
    """Build against staged evidence without touching the currently served site."""
    generation.mkdir(parents=True, exist_ok=True)
    for filename in ("go2_back_qr_training_metrics.json", "go2_legged_loco_metrics.json"):
        source = artifacts / filename
        destination = generation / filename
        if not destination.is_file():
            if not source.is_file():
                raise RuntimeError(f"missing dashboard input: {source}")
            shutil.copy2(source, destination)
    normalize_onnx_manifest(generation, artifacts, models)
    builder_path = project_root / "scripts" / "build_go2_back_qr_dashboard.py"
    spec = importlib.util.spec_from_file_location("staged_go2_dashboard_builder", builder_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import dashboard builder: {builder_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ARTIFACTS = generation
    module.main()


def validate_onnx(generation: Path, errors: list[str]) -> None:
    manifest_path = generation / "go2_back_qr_onnx_models.json"
    if not manifest_path.is_file():
        errors.append(f"missing ONNX manifest: {manifest_path}")
        return
    manifest = read_json(manifest_path)
    entries = manifest.get("models", [])
    by_algorithm = {
        entry.get("algorithm"): entry
        for entry in entries
        if isinstance(entry, dict) and entry.get("algorithm") in ALGORITHMS
    }
    if len(entries) != len(ALGORITHMS) or set(by_algorithm) != set(ALGORITHMS):
        errors.append("staged ONNX manifest must contain exactly PPO, DDPG and SAC")
        return
    import numpy as np
    import onnxruntime as ort
    preflight_report = read_json(generation / "go2_back_qr_suite_preflight.json")

    for algorithm in ALGORITHMS:
        onnx_path = generation / "onnx_go2" / f"{algorithm}_go2_back_qr.onnx"
        entry = by_algorithm[algorithm]
        if not onnx_path.is_file() or onnx_path.stat().st_size == 0:
            errors.append(f"missing or empty staged ONNX: {onnx_path}")
            continue
        if entry.get("input", {}).get("shape") != ["batch", len(OBSERVATION_NAMES)]:
            errors.append(f"{algorithm} ONNX manifest input is not 7D")
        if entry.get("output", {}).get("shape") != ["batch", 2]:
            errors.append(f"{algorithm} ONNX manifest output is not 2D")
        if entry.get("source_model_sha256") != preflight_report.get("landing_model_sha256", {}).get(algorithm):
            errors.append(f"{algorithm} ONNX source hash is not the accepted SB3 model hash")
        if entry.get("onnx_sha256") != sha256_file(onnx_path):
            errors.append(f"{algorithm} staged ONNX hash differs from its export manifest")
        difference = entry.get("validation_max_abs_action_error")
        if not finite_number(difference) or float(difference) > 2e-5:
            errors.append(f"{algorithm} ONNX parity error is missing or above 2e-5")
        try:
            session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            input_meta = session.get_inputs()[0]
            output = session.run(None, {input_meta.name: np.zeros((1, 7), dtype=np.float32)})[0]
            if tuple(output.shape) != (1, 2):
                errors.append(f"{algorithm} ONNX runtime output shape is {output.shape}, expected (1, 2)")
        except Exception as error:  # pragma: no cover - runtime supplies backend-specific errors
            errors.append(f"cannot execute staged {algorithm} ONNX: {error}")


def validate_demo(generation: Path, algorithm: str, difficulty: str, errors: list[str]) -> None:
    import cv2

    stem = demo_stem(algorithm, difficulty)
    video = generation / f"{stem}.mp4"
    snapshot = generation / f"{stem}.png"
    log = generation / f"{stem}.csv"
    for path in (video, snapshot, log):
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing or empty {algorithm}/{difficulty} artifact: {path}")
    if not log.is_file() or log.stat().st_size == 0:
        return
    try:
        with log.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or ())
            rows = list(reader)
    except (OSError, csv.Error) as error:
        errors.append(f"cannot read {algorithm}/{difficulty} CSV: {error}")
        return
    missing = sorted(INFERENCE_CSV_FIELDS - fields)
    if missing:
        errors.append(f"{algorithm}/{difficulty} CSV missing fields: {', '.join(missing)}")
    legacy = sorted(LEGACY_CSV_FIELDS & fields)
    if legacy:
        errors.append(f"{algorithm}/{difficulty} CSV contains legacy unprefixed fields: {', '.join(legacy)}")
    if not rows:
        errors.append(f"{algorithm}/{difficulty} CSV has no frames")
    elif not missing:
        try:
            numeric_fields = INFERENCE_CSV_FIELDS - {"onnx_provider"}
            numeric_values = {
                field: [float(row[field]) for row in rows]
                for field in numeric_fields
            }
            nonfinite = sorted(
                field
                for field, values in numeric_values.items()
                if any(not math.isfinite(value) for value in values)
            )
            if nonfinite:
                errors.append(f"{algorithm}/{difficulty} CSV has non-finite fields: {', '.join(nonfinite)}")
            contacts = max(numeric_values["offline_sim_landing_skid_contacts"])
            penetration = max(numeric_values["offline_sim_max_contact_penetration_m"])
            visual_contact_plane_error = max(
                abs(value)
                for value in numeric_values["offline_sim_visual_contact_plane_error_m"]
            )
            root_wrench = max(abs(value) for value in numeric_values["offline_sim_go2_root_wrench_max_abs"])
            if contacts < 2.0:
                errors.append(
                    f"{algorithm}/{difficulty} recording never shows both stock skid rails on the deck"
                )
            if penetration > MAX_OFFLINE_SIM_PENETRATION_M:
                errors.append(f"{algorithm}/{difficulty} recording penetration exceeds 2 mm")
            if visual_contact_plane_error > MAX_VISUAL_CONTACT_PLANE_ERROR_M:
                errors.append(
                    f"{algorithm}/{difficulty} rendered skid/contact sole mismatch exceeds 1 mm"
                )
            if root_wrench != 0.0:
                errors.append(f"{algorithm}/{difficulty} recording uses nonzero Go2 root wrench")
            drone_pixels = numeric_values["third_person_drone_pixels"]
            if any(value < MIN_THIRD_PERSON_DRONE_PIXELS for value in drone_pixels):
                errors.append(
                    f"{algorithm}/{difficulty} third-person X500 visibility drops below "
                    f"{MIN_THIRD_PERSON_DRONE_PIXELS:.0f} segmentation pixels"
                )
            down_nonblack = numeric_values["down_view_nonblack_fraction"]
            if any(value < MIN_RUNTIME_DOWN_VIEW_NONBLACK_FRACTION for value in down_nonblack):
                errors.append(
                    f"{algorithm}/{difficulty} runtime down-view nonblack fraction drops below "
                    f"{MIN_RUNTIME_DOWN_VIEW_NONBLACK_FRACTION:.2f}"
                )
            down_luma_std = numeric_values["down_view_luma_std"]
            if any(value < MIN_RUNTIME_DOWN_VIEW_LUMA_STD for value in down_luma_std):
                errors.append(
                    f"{algorithm}/{difficulty} runtime down-view luma std drops below "
                    f"{MIN_RUNTIME_DOWN_VIEW_LUMA_STD:.1f}"
                )
            state_matches = numeric_values["dual_view_state_match"]
            if any(value != 1.0 for value in state_matches):
                errors.append(f"{algorithm}/{difficulty} dual views do not share one simulator state")
            visibility_sampled = numeric_values["third_person_visibility_sampled"]
            if any(value not in (0.0, 1.0) for value in visibility_sampled):
                errors.append(f"{algorithm}/{difficulty} visibility sample flags are not binary")
            sample_indices = [index for index, value in enumerate(visibility_sampled) if value == 1.0]
            if not sample_indices or sample_indices[0] != 0:
                errors.append(f"{algorithm}/{difficulty} has no initial X500 segmentation proof")
            elif (
                any(
                    current - previous > SEGMENTATION_SAMPLE_INTERVAL_FRAMES
                    for previous, current in zip(sample_indices, sample_indices[1:])
                )
                or len(rows) - 1 - sample_indices[-1] >= SEGMENTATION_SAMPLE_INTERVAL_FRAMES
            ):
                errors.append(
                    f"{algorithm}/{difficulty} X500 segmentation proof gap exceeds "
                    f"{SEGMENTATION_SAMPLE_INTERVAL_FRAMES} frames"
                )
            projection_visible = numeric_values["third_person_projection_visible"]
            if any(value != 1.0 for value in projection_visible):
                errors.append(f"{algorithm}/{difficulty} projected X500 locator leaves the third-person view")
            if any(row["onnx_provider"] != "CPUExecutionProvider" for row in rows):
                errors.append(f"{algorithm}/{difficulty} CSV has an unexpected ONNX provider")
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"{algorithm}/{difficulty} CSV contains invalid offline_sim diagnostics: {error}")
    if video.is_file() and video.stat().st_size:
        capture = cv2.VideoCapture(str(video))
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        frame_rate = float(capture.get(cv2.CAP_PROP_FPS))
        ok, frame = capture.read()
        last_ok = ok if frame_count == 1 else False
        if frame_count > 1:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
            last_ok, _ = capture.read()
        down_view_issues = validate_down_view_video(capture, frame_count)
        capture.release()
        if not ok or frame is None:
            errors.append(f"{algorithm}/{difficulty} MP4 is not decodable")
        elif frame.shape[:2] != (720, 1920):
            errors.append(f"{algorithm}/{difficulty} MP4 is {frame.shape[1]}x{frame.shape[0]}, expected 1920x720")
        if not last_ok:
            errors.append(f"{algorithm}/{difficulty} MP4 final frame is not decodable")
        if not math.isfinite(frame_rate) or abs(frame_rate - 30.0) > 0.05:
            errors.append(f"{algorithm}/{difficulty} MP4 frame rate is {frame_rate}, expected 30")
        if rows and abs(frame_count - len(rows)) > 1:
            errors.append(
                f"{algorithm}/{difficulty} MP4/CSV frame mismatch: {frame_count} vs {len(rows)}"
            )
        errors.extend(
            f"{algorithm}/{difficulty} {issue}"
            for issue in down_view_issues
        )
    if snapshot.is_file() and snapshot.stat().st_size:
        frame = cv2.imread(str(snapshot), cv2.IMREAD_COLOR)
        if frame is None:
            errors.append(f"{algorithm}/{difficulty} PNG is not decodable")
        elif frame.shape[:2] != (720, 1920):
            errors.append(f"{algorithm}/{difficulty} PNG is {frame.shape[1]}x{frame.shape[0]}, expected 1920x720")
    receipt_path = generation / f"{stem}.receipt.json"
    if not receipt_path.is_file():
        errors.append(f"missing stable-landing receipt: {receipt_path}")
    else:
        receipt = read_json(receipt_path)
        if receipt.get("status") != "stable_landing":
            errors.append(f"{algorithm}/{difficulty} receipt does not confirm stable landing")
        if receipt.get("algorithm") != algorithm or receipt.get("difficulty") != difficulty:
            errors.append(f"{algorithm}/{difficulty} receipt identity mismatch")
        if receipt.get("seed") != VIDEO_SEEDS[(algorithm, difficulty)]:
            errors.append(f"{algorithm}/{difficulty} receipt seed mismatch")
        for extension, path in (("mp4", video), ("png", snapshot), ("csv", log)):
            if path.is_file() and receipt.get("artifact_sha256", {}).get(extension) != sha256_file(path):
                errors.append(f"{algorithm}/{difficulty} {extension} changed after successful inference")
        onnx_path = generation / "onnx_go2" / f"{algorithm}_go2_back_qr.onnx"
        locomotion_path = generation / "model_inputs" / "go2_legged_loco_ppo.zip"
        if onnx_path.is_file() and receipt.get("onnx_sha256") != sha256_file(onnx_path):
            errors.append(f"{algorithm}/{difficulty} receipt ONNX hash mismatch")
        if locomotion_path.is_file() and receipt.get("locomotion_model_sha256") != sha256_file(locomotion_path):
            errors.append(f"{algorithm}/{difficulty} receipt locomotion hash mismatch")


def record_demo(generation: Path, algorithm: str, difficulty: str, seed: int) -> Path:
    """Write a receipt only after inference returned a stable-landing exit code."""
    if (algorithm, difficulty) not in VIDEO_SEEDS:
        raise RuntimeError(f"unknown demonstration identity: {algorithm}/{difficulty}")
    if seed != VIDEO_SEEDS[(algorithm, difficulty)]:
        raise RuntimeError(f"unexpected demonstration seed for {algorithm}/{difficulty}: {seed}")
    stem = demo_stem(algorithm, difficulty)
    assets = {extension: generation / f"{stem}.{extension}" for extension in ("mp4", "png", "csv")}
    onnx_path = generation / "onnx_go2" / f"{algorithm}_go2_back_qr.onnx"
    locomotion_path = generation / "model_inputs" / "go2_legged_loco_ppo.zip"
    for path in (*assets.values(), onnx_path, locomotion_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"cannot record successful demonstration; missing file: {path}")
    receipt = {
        "status": "stable_landing",
        "algorithm": algorithm,
        "difficulty": difficulty,
        "seed": seed,
        "artifact_sha256": {extension: sha256_file(path) for extension, path in assets.items()},
        "onnx_sha256": sha256_file(onnx_path),
        "locomotion_model_sha256": sha256_file(locomotion_path),
    }
    path = generation / f"{stem}.receipt.json"
    write_json_atomic(path, receipt)
    return path


def publishable_files(generation: Path) -> list[Path]:
    files = [
        generation / "go2_back_qr_landing_dashboard.html",
        generation / "go2_back_qr_onnx_models.json",
    ]
    files.extend(
        generation / f"{demo_stem(algorithm, difficulty)}.{extension}"
        for algorithm in ALGORITHMS
        for difficulty in DIFFICULTIES
        for extension in ("mp4", "png", "csv", "receipt.json")
    )
    files.extend(generation / "onnx_go2" / f"{algorithm}_go2_back_qr.onnx" for algorithm in ALGORITHMS)
    return files


def dashboard_content_issues(document: str) -> list[str]:
    """Return human-readable dashboard contract violations.

    This is intentionally independent of the video/ONNX generation checks so
    the Korean documentation contract can be covered with small unit tests.
    Semantic ``data-*`` markers protect meaning across harmless prose/layout
    edits, while the visible Korean fragments ensure the contract is explained
    to a reader rather than encoded only as hidden metadata.
    """
    issues: list[str] = []
    for required in (
        'lang="ko"',
        "offline_sim_*",
        "착륙다리 touch/load/contact 센서는 존재하지",
        "tex-mml-chtml.js",
    ):
        if required not in document:
            issues.append(f"dashboard is missing required Korean/sensor-only content: {required}")

    for label, marker in DASHBOARD_INPUT_GUIDE_MARKERS:
        if marker not in document:
            issues.append(f"dashboard is missing {label} contract marker: {marker}")
    for observation_name in OBSERVATION_NAMES:
        marker = f'data-observation="{observation_name}"'
        if marker not in document:
            issues.append(f"dashboard 7D input guide is missing observation row: {observation_name}")
    for required_text in DASHBOARD_INPUT_GUIDE_VISIBLE_TEXT:
        if required_text not in document:
            issues.append(f"dashboard input guide is missing visible explanation: {required_text}")

    # Match the retired final-alignment multiplier as an expression, not
    # the bare decimal: a legitimate measured metric can round to 0.42.
    for forbidden in ("c_gear/4", "v_pad,x", "go2_assist_force_n", r"\leftarrow0.42"):
        if forbidden in document:
            issues.append(f"dashboard contains stale privileged/slowdown content: {forbidden}")
    for algorithm in ALGORITHMS:
        for difficulty in DIFFICULTIES:
            if f"{demo_stem(algorithm, difficulty)}.mp4" not in document:
                issues.append(f"dashboard does not link {algorithm}/{difficulty} MP4")
    return issues


def validate_generation(generation: Path) -> dict[str, str]:
    errors: list[str] = []
    validate_onnx(generation, errors)
    for algorithm in ALGORITHMS:
        for difficulty in DIFFICULTIES:
            validate_demo(generation, algorithm, difficulty, errors)
    dashboard = generation / "go2_back_qr_landing_dashboard.html"
    if not dashboard.is_file() or dashboard.stat().st_size == 0:
        errors.append(f"missing or empty Korean dashboard: {dashboard}")
    else:
        document = dashboard.read_text(encoding="utf-8")
        errors.extend(dashboard_content_issues(document))
    for path in publishable_files(generation):
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing publishable file: {path}")
    if errors:
        raise RuntimeError("generation validation failed:\n- " + "\n- ".join(errors))
    return {
        str(path.relative_to(generation)): sha256_file(path)
        for path in publishable_files(generation)
    }


def finalize_generation(generation: Path) -> Path:
    hashes = validate_generation(generation)
    landing = read_json(generation / "go2_back_qr_training_metrics.json")
    preflight_report = read_json(generation / "go2_back_qr_suite_preflight.json")
    if preflight_report.get("status") != "passed":
        raise RuntimeError("staged preflight record is not passed")
    record = {
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "algorithms": list(ALGORITHMS),
        "difficulties": list(DIFFICULTIES),
        "video_count": 9,
        "observation_names": list(OBSERVATION_NAMES),
        "landing_model_sha256": landing.get("promotion", {}).get("canonical_model_sha256", {}),
        "locomotion_model_sha256": landing.get("locomotion_model_sha256"),
        "input_snapshot": preflight_report,
        "published_file_sha256": hashes,
    }
    path = generation / "go2_back_qr_suite_generation.json"
    write_json_atomic(path, record)
    return path


def transactional_replace(replacements: list[tuple[Path, Path]]) -> None:
    """Publish a complete flat artifact set with rollback on any failed swap."""
    targets = [target for _, target in replacements]
    if len(targets) != len(set(targets)):
        raise ValueError("duplicate publication target")
    token = uuid.uuid4().hex
    pending: dict[Path, Path] = {}
    backups: dict[Path, Path] = {}
    installed: list[Path] = []
    succeeded = False
    try:
        for source, target in replacements:
            if not source.is_file():
                raise FileNotFoundError(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            staged = target.with_name(f".{target.name}.{token}.pending")
            shutil.copy2(source, staged)
            pending[target] = staged
        for _, target in replacements:
            if target.exists() or target.is_symlink():
                backup = target.with_name(f".{target.name}.{token}.backup")
                if not target.is_file():
                    raise RuntimeError(f"publication target is not a file: {target}")
                shutil.copy2(target, backup)
                backups[target] = backup
        for _, target in replacements:
            pending[target].replace(target)
            installed.append(target)
        succeeded = True
    except BaseException as publish_error:
        rollback_errors: list[str] = []
        for target in reversed(installed):
            try:
                backup = backups.get(target)
                if backup is None:
                    target.unlink(missing_ok=True)
                else:
                    backup.replace(target)
            except OSError as error:
                rollback_errors.append(f"restore {target}: {error}")
        if rollback_errors:
            raise RuntimeError(
                f"publication failed ({publish_error}) and rollback was incomplete: "
                + "; ".join(rollback_errors)
            ) from publish_error
        for backup in backups.values():
            backup.unlink(missing_ok=True)
        raise
    finally:
        for staged in pending.values():
            staged.unlink(missing_ok=True)
        if succeeded:
            for backup in backups.values():
                backup.unlink(missing_ok=True)


def publish_generation(generation: Path, artifacts: Path, models: Path) -> None:
    hashes = validate_generation(generation)
    record_path = generation / "go2_back_qr_suite_generation.json"
    record = read_json(record_path)
    if record.get("status") != "complete" or record.get("video_count") != 9:
        raise RuntimeError("generation record is not complete")
    if record.get("published_file_sha256") != hashes:
        raise RuntimeError("generation changed after final validation")
    input_snapshot = record.get("input_snapshot", {})
    if not isinstance(input_snapshot, dict) or input_snapshot.get("status") != "passed":
        raise RuntimeError("generation has no passed input snapshot")
    for algorithm in ALGORITHMS:
        current_model = models / f"{algorithm}_go2_back_qr.zip"
        expected_hash = input_snapshot.get("landing_model_sha256", {}).get(algorithm)
        if not current_model.is_file() or sha256_file(current_model) != expected_hash:
            raise RuntimeError(f"{algorithm} canonical changed after capture; publication refused")
    current_locomotion = Path(str(input_snapshot.get("locomotion_model", "")))
    if not current_locomotion.is_file() or sha256_file(current_locomotion) != input_snapshot.get("locomotion_model_sha256"):
        raise RuntimeError("low-level locomotion model changed after capture; publication refused")
    for filename, hash_key in (
        ("go2_back_qr_training_metrics.json", "landing_metrics_sha256"),
        ("go2_legged_loco_metrics.json", "locomotion_metrics_sha256"),
    ):
        current_metrics = artifacts / filename
        if not current_metrics.is_file() or sha256_file(current_metrics) != input_snapshot.get(hash_key):
            raise RuntimeError(f"{filename} changed after capture; publication refused")

    replacements: list[tuple[Path, Path]] = [
        (
            generation / "go2_back_qr_onnx_models.json",
            artifacts / "go2_back_qr_onnx_models.json",
        ),
    ]
    replacements.extend(
        (
            generation / f"{demo_stem(algorithm, difficulty)}.{extension}",
            artifacts / f"{demo_stem(algorithm, difficulty)}.{extension}",
        )
        for algorithm in ALGORITHMS
        for difficulty in DIFFICULTIES
        for extension in ("mp4", "png", "csv", "receipt.json")
    )
    for algorithm in ALGORITHMS:
        source = generation / "onnx_go2" / f"{algorithm}_go2_back_qr.onnx"
        replacements.append((source, artifacts / "onnx_go2" / source.name))
        replacements.append((source, models / "onnx_go2" / source.name))
    # Publish the HTML only after every referenced immutable evidence file is
    # present. Existing readers never observe a missing target during swaps.
    replacements.append(
        (
            generation / "go2_back_qr_landing_dashboard.html",
            artifacts / "go2_back_qr_landing_dashboard.html",
        )
    )
    # The generation record is the commit marker and is installed last.
    replacements.append((record_path, artifacts / record_path.name))
    artifacts.mkdir(parents=True, exist_ok=True)
    with (artifacts / ".go2_suite_publish.lock").open("a+") as publish_lock:
        fcntl.flock(publish_lock.fileno(), fcntl.LOCK_EX)
        transactional_replace(replacements)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    preflight_parser.add_argument("--artifacts-dir", type=Path, required=True)
    preflight_parser.add_argument("--models-dir", type=Path, required=True)
    preflight_parser.add_argument("--locomotion-model", type=Path, required=True)

    snapshot_parser = subparsers.add_parser("snapshot")
    snapshot_parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    snapshot_parser.add_argument("--generation-dir", type=Path, required=True)
    snapshot_parser.add_argument("--artifacts-dir", type=Path, required=True)
    snapshot_parser.add_argument("--models-dir", type=Path, required=True)
    snapshot_parser.add_argument("--locomotion-model", type=Path, required=True)

    build_parser = subparsers.add_parser("build-dashboard")
    build_parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    build_parser.add_argument("--generation-dir", type=Path, required=True)
    build_parser.add_argument("--artifacts-dir", type=Path, required=True)
    build_parser.add_argument("--models-dir", type=Path, required=True)

    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--generation-dir", type=Path, required=True)

    receipt_parser = subparsers.add_parser("record-demo")
    receipt_parser.add_argument("--generation-dir", type=Path, required=True)
    receipt_parser.add_argument("--algorithm", choices=ALGORITHMS, required=True)
    receipt_parser.add_argument("--difficulty", choices=DIFFICULTIES, required=True)
    receipt_parser.add_argument("--seed", type=int, required=True)

    publish_parser = subparsers.add_parser("publish")
    publish_parser.add_argument("--generation-dir", type=Path, required=True)
    publish_parser.add_argument("--artifacts-dir", type=Path, required=True)
    publish_parser.add_argument("--models-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "preflight":
        report = preflight(
            args.project_root.resolve(),
            args.artifacts_dir.resolve(),
            args.models_dir.resolve(),
            args.locomotion_model.resolve(),
        )
        print(json.dumps(report, indent=2), flush=True)
    elif args.command == "snapshot":
        report = snapshot_inputs(
            args.project_root.resolve(),
            args.artifacts_dir.resolve(),
            args.models_dir.resolve(),
            args.locomotion_model.resolve(),
            args.generation_dir.resolve(),
        )
        print(json.dumps(report, indent=2), flush=True)
    elif args.command == "build-dashboard":
        build_dashboard(
            args.project_root.resolve(),
            args.generation_dir.resolve(),
            args.artifacts_dir.resolve(),
            args.models_dir.resolve(),
        )
    elif args.command == "finalize":
        record = finalize_generation(args.generation_dir.resolve())
        print(f"Validated complete 9-video generation: {record}", flush=True)
    elif args.command == "record-demo":
        receipt = record_demo(
            args.generation_dir.resolve(), args.algorithm, args.difficulty, args.seed
        )
        print(f"Recorded stable-landing receipt: {receipt}", flush=True)
    elif args.command == "publish":
        publish_generation(
            args.generation_dir.resolve(),
            args.artifacts_dir.resolve(),
            args.models_dir.resolve(),
        )
        print("Published complete Go2 QR generation transactionally", flush=True)
    else:  # pragma: no cover - argparse guarantees a known command
        raise AssertionError(args.command)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        raise SystemExit(str(error)) from None
