#!/usr/bin/env python3
"""Publish nine verified moving-landings on the physical discrete gravel road.

The three difficulty stages are intentionally identical except for the Go2
forward-speed command.  A recording is rejected if Go2 falls, leaves the
finite collision road, loses the one-second motion-window requirement, uses
any non-sole Go2 leg collision as terrain support, or if the X500 never
produces a two-skid physical landing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from landing_rl.go2_qr_environment import GRAVEL_MOTION_GUARD_START_S, GRAVEL_MIN_WINDOW_SPEED_MPS, GRAVEL_ROUTE_SPEED_MPS
from landing_rl.go2_terrain import GRAVEL_ROCK_BANDS, GRAVEL_SLOPE_GRADE, gravel_rock_specs, terrain_metadata


ARTIFACTS = PROJECT_ROOT / "artifacts" / "rl_training"
ALGORITHMS = ("ppo", "ddpg", "sac")
STAGES = (("easy", "초급"), ("medium", "중급"), ("hard", "고급"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def summarize_csv(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "sim_time_s", "qr_error_m", "altitude_m", "offline_sim_landing_skid_contacts",
        "offline_sim_max_contact_penetration_m", "offline_sim_go2_root_wrench_max_abs",
        "offline_sim_go2_tilt_deg", "offline_sim_terrain_course_inside",
        "offline_sim_terrain_boundary_clearance_m", "offline_sim_go2_motion_ok",
        "offline_sim_go2_motion_window_speed_mps", "offline_sim_go2_motion_violation",
        "offline_sim_go2_speed_mps", "offline_sim_go2_path_distance_m",
        "offline_sim_go2_sole_contacts", "offline_sim_go2_sole_normal_force_n",
        "offline_sim_go2_sole_contact_peak", "offline_sim_go2_nonsole_terrain_contacts",
        "offline_sim_go2_nonsole_terrain_violation",
    }
    if not rows or not required.issubset(set(rows[0])):
        raise RuntimeError(f"discrete-gravel CSV lacks required diagnostics: {path}")

    numeric = lambda name: [float(row[name]) for row in rows]
    contacts = numeric("offline_sim_landing_skid_contacts")
    penetration = numeric("offline_sim_max_contact_penetration_m")
    tilt = numeric("offline_sim_go2_tilt_deg")
    clearance = numeric("offline_sim_terrain_boundary_clearance_m")
    inside = numeric("offline_sim_terrain_course_inside")
    wrench = numeric("offline_sim_go2_root_wrench_max_abs")
    violation = numeric("offline_sim_go2_motion_violation")
    sole_contacts = numeric("offline_sim_go2_sole_contacts")
    sole_force = numeric("offline_sim_go2_sole_normal_force_n")
    sole_peak = numeric("offline_sim_go2_sole_contact_peak")
    nonsole_contacts = numeric("offline_sim_go2_nonsole_terrain_contacts")
    nonsole_violation = numeric("offline_sim_go2_nonsole_terrain_violation")
    moving_rows = [
        row for row in rows
        if float(row["sim_time_s"]) >= GRAVEL_MOTION_GUARD_START_S
    ]
    if max(contacts) < 2.0:
        raise RuntimeError(f"two visible X500 skid rails never land: {path}")
    if max(penetration) > 0.0021:
        raise RuntimeError(f"landing penetration exceeds 2.1 mm: {path}")
    if max(tilt) > 35.0:
        raise RuntimeError(f"Go2 tilt exceeds 35 degrees: {path}")
    if min(inside) < 1.0 or min(clearance) < 0.0:
        raise RuntimeError(f"Go2 or QR deck leaves the finite gravel road: {path}")
    if max(abs(value) for value in wrench) != 0.0:
        raise RuntimeError(f"Go2 root wrench is nonzero: {path}")
    if max(violation) > 0.0:
        raise RuntimeError(f"Go2 stopped during a gravel-road landing: {path}")
    if max(nonsole_contacts) > 0.0 or max(nonsole_violation) > 0.0:
        raise RuntimeError(f"Go2 used a non-sole leg collision on terrain: {path}")
    if max(sole_contacts) < 2.0 or max(sole_peak) < 2.0 or max(sole_force) <= 1.0:
        raise RuntimeError(f"Go2 never established physical rubber-sole support: {path}")
    if not moving_rows:
        raise RuntimeError(f"recording ends before gravel motion guard: {path}")
    window_speeds = [float(row["offline_sim_go2_motion_window_speed_mps"]) for row in moving_rows]
    if min(window_speeds) < GRAVEL_MIN_WINDOW_SPEED_MPS:
        raise RuntimeError(f"Go2 motion window falls below {GRAVEL_MIN_WINDOW_SPEED_MPS:.2f} m/s: {path}")
    return {
        "frames": float(len(rows)),
        "duration_s": float(rows[-1]["sim_time_s"]),
        "terminal_qr_error_m": float(rows[-1]["qr_error_m"]),
        "terminal_relative_altitude_m": float(rows[-1]["altitude_m"]),
        "max_skid_contacts": max(contacts),
        "max_penetration_m": max(penetration),
        "max_go2_tilt_deg": max(tilt),
        "min_terrain_boundary_clearance_m": min(clearance),
        "terminal_go2_path_distance_m": float(rows[-1]["offline_sim_go2_path_distance_m"]),
        "terminal_go2_speed_mps": float(rows[-1]["offline_sim_go2_speed_mps"]),
        "terminal_motion_window_speed_mps": float(rows[-1]["offline_sim_go2_motion_window_speed_mps"]),
        "min_motion_window_speed_mps": min(window_speeds),
        "max_go2_sole_contacts": max(sole_contacts),
        "max_go2_sole_normal_force_n": max(sole_force),
        "max_go2_nonsole_terrain_contacts": max(nonsole_contacts),
    }


def assert_video(path: Path) -> None:
    capture = cv2.VideoCapture(str(path))
    frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    first_ok, first = capture.read()
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count - 1))
    last_ok, _ = capture.read()
    capture.release()
    if not first_ok or first is None or not last_ok:
        raise RuntimeError(f"MP4 is not decodable end-to-end: {path}")
    if first.shape[:2] != (720, 1920) or abs(fps - 30.0) > 0.05:
        raise RuntimeError(f"MP4 contract mismatch: {path}, {first.shape}, {fps} fps")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    parser.add_argument("--locomotion-model", type=Path, default=PROJECT_ROOT / "models" / "go2_legged_loco_ppo.zip")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = args.artifacts.resolve()
    locomotion_model = args.locomotion_model.resolve()
    if not locomotion_model.is_file():
        raise RuntimeError(f"missing learned Go2 PPO: {locomotion_model}")
    onnx_paths = {algorithm: artifacts / "onnx_go2" / f"{algorithm}_go2_back_qr.onnx" for algorithm in ALGORITHMS}
    for path in onnx_paths.values():
        if not path.is_file():
            raise RuntimeError(f"missing landing ONNX model: {path}")

    generation = artifacts / f".go2_discrete_gravel_suite.{uuid.uuid4().hex[:8]}"
    generation.mkdir(parents=True, exist_ok=False)
    try:
        inputs = generation / "model_inputs"
        inputs.mkdir()
        staged_loco = inputs / locomotion_model.name
        shutil.copy2(locomotion_model, staged_loco)
        staged_onnx: dict[str, Path] = {}
        for algorithm, source in onnx_paths.items():
            target = inputs / source.name
            shutil.copy2(source, target)
            staged_onnx[algorithm] = target

        records: list[dict[str, Any]] = []
        for algorithm_index, algorithm in enumerate(ALGORITHMS):
            for stage_index, (difficulty, korean_name) in enumerate(STAGES):
                stem = f"{algorithm}_go2_discrete_gravel_{difficulty}_follow"
                video = generation / f"{stem}.mp4"
                snapshot = generation / f"{stem}.png"
                log = generation / f"{stem}.csv"
                # One declared fixed seed per method/stage: a failed result is
                # a failed result, never hidden by seed-shopping.
                seed = 20262301 + 101 * algorithm_index + 13 * stage_index
                command = [
                    sys.executable, "-B", "-m", "landing_rl.go2_onnx_inference",
                    "--onnx-model", str(staged_onnx[algorithm]),
                    "--locomotion-model", str(staged_loco),
                    "--difficulty", difficulty, "--terrain-task", "gravel", "--seed", str(seed),
                    "--video-file", str(video), "--snapshot-file", str(snapshot), "--log-file", str(log),
                    "--fps", "30",
                ]
                completed = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True)
                if completed.returncode != 0:
                    raise RuntimeError(f"inference failed for {algorithm}/{difficulty}:\n{completed.stdout}\n{completed.stderr}")
                for artifact in (video, snapshot, log):
                    if not artifact.is_file() or artifact.stat().st_size == 0:
                        raise RuntimeError(f"missing generated artifact: {artifact}")
                assert_video(video)
                summary = summarize_csv(log)
                receipt_path = generation / f"{stem}.receipt.json"
                receipt = {
                    "status": "stable_moving_landing",
                    "algorithm": algorithm,
                    "difficulty": difficulty,
                    "seed": seed,
                    "terrain": terrain_metadata("gravel"),
                    "command_speed_mps": GRAVEL_ROUTE_SPEED_MPS[difficulty],
                    "landing_gate": "two physical X500 skids, Go2/QR deck inside finite road, Go2 fall=0, root wrench=0, 1 s Go2 motion window >= 0.12 m/s; stopped Go2 invalidates landing",
                    "onnx_sha256": sha256_file(staged_onnx[algorithm]),
                    "locomotion_model": locomotion_model.name,
                    "locomotion_model_sha256": sha256_file(staged_loco),
                    "artifact_sha256": {suffix: sha256_file(path) for suffix, path in (("mp4", video), ("png", snapshot), ("csv", log))},
                    "summary": summary,
                }
                write_json(receipt_path, receipt)
                records.append({
                    "algorithm": algorithm,
                    "difficulty": difficulty,
                    "korean_name": korean_name,
                    "seed": seed,
                    "video": video.name,
                    "snapshot": snapshot.name,
                    "csv": log.name,
                    "receipt": receipt_path.name,
                    "command_speed_mps": GRAVEL_ROUTE_SPEED_MPS[difficulty],
                    "summary": summary,
                    "onnx_sha256": receipt["onnx_sha256"],
                    "locomotion_model": locomotion_model.name,
                    "locomotion_model_sha256": receipt["locomotion_model_sha256"],
                })

        report = {
            "status": "passed",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "terrain": terrain_metadata("gravel"),
            "physical_terrain": {
                "individual_static_collision_rocks": len(gravel_rock_specs()),
                "rock_bands": [list(band) for band in GRAVEL_ROCK_BANDS],
                "soil_grade_percent": 100.0 * GRAVEL_SLOPE_GRADE,
                "description": "Individual rounded static MuJoCo collision stones embedded into a gently undulating 0.8% compacted-soil heightfield.",
            },
            "difficulty_contract": "Beginner, intermediate and advanced use the exact same road, drone setup and landing gate. Only Go2 commanded forward speed changes: 0.58 / 0.75 / 0.92 m/s.",
            "locomotion": {
                "model": locomotion_model.name,
                "sha256": sha256_file(staged_loco),
                "training_note": "A discrete-gravel fine-tuning candidate was retained as an audit artifact but rejected because it did not preserve all three commanded-speed replays. Published demonstrations use the learned PPO that passes every declared combined replay.",
            },
            "algorithms": list(ALGORITHMS),
            "demonstrations": records,
        }
        write_json(generation / "go2_discrete_gravel_landing_suite.json", report)

        for path in generation.iterdir():
            if path.name == "model_inputs":
                continue
            destination = artifacts / path.name
            if destination.exists():
                destination.unlink()
            shutil.move(str(path), str(destination))
        print(f"Published {len(records)} verified discrete-gravel moving landings")
    finally:
        shutil.rmtree(generation, ignore_errors=True)


if __name__ == "__main__":
    main()
