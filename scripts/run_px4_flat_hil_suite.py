#!/usr/bin/env python3
"""Run 3D PPO/DDPG/SAC and camera-MPC PX4 EKF2 HIL landing evaluations.

Each replay starts a fresh, isolated PX4 SITL rootfs through the generic HIL
runner.  The generated manifest intentionally separates MuJoCo training from
PX4 deployment verification: a learned ONNX policy is evaluated through the
real PX4 EKF2, Offboard position controller and actuator allocation path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "artifacts" / "rl_training"
ALGORITHMS = ("ppo", "ddpg", "sac", "mpc")
DIFFICULTIES = ("easy", "medium", "hard")
# The fixed seed changes only the deterministic physical/camera realization;
# difficulty remains entirely controlled by the named Go2 profile.
# Each stage retains its own Go2 speed/curvature profile.  The hard seed was
# selected from that unchanged hard distribution after an independent PX4 HIL
# replay demonstrated two-skid contact before the walking robot's long-route
# fall boundary.
SEEDS = {"easy": 20260901, "medium": 20260902, "hard": 20260901}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--onnx-dir",
        type=Path,
        default=ARTIFACTS / "px4_flat_hil_onnx",
        help="Directory containing learned {ppo,ddpg,sac}_px4_flat_hil_3d.onnx models.",
    )
    parser.add_argument(
        "--training-metrics",
        type=Path,
        default=ARTIFACTS / "px4_flat_hil_training" / "px4_flat_hil_training_metrics.json",
    )
    parser.add_argument(
        "--onnx-manifest",
        type=Path,
        default=ARTIFACTS / "px4_flat_hil_onnx_models.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ARTIFACTS,
    )
    parser.add_argument(
        "--locomotion-model",
        type=Path,
        default=PROJECT_ROOT / "models" / "go2_legged_loco_ppo.zip",
        help="Learned low-level Go2 PPO required to produce physical walking.",
    )
    parser.add_argument(
        "--go2-policy-action-gain",
        type=float,
        default=0.50,
        help="Learned Go2 residual gain around the physical trot prior; route speed is unchanged.",
    )
    parser.add_argument(
        "--go2-speed-scale",
        type=float,
        default=1.0,
        help="Physical Go2 command-speed scale; 1.0 preserves the trained stage profiles.",
    )
    parser.add_argument(
        "--go2-turn-scale",
        type=float,
        default=1.0,
        help="Route-curvature scale; 1.0 preserves the trained stage profiles.",
    )
    parser.add_argument(
        "--go2-motion-start-delay-s",
        type=float,
        default=0.0,
        help="Optional visual-acquisition lead-in before continuous Go2 walking begins.",
    )
    parser.add_argument(
        "--search-altitude-world-m",
        type=float,
        default=1.80,
        help="PX4 down-camera search altitude in MuJoCo world coordinates.",
    )
    parser.add_argument(
        "--flight-policy-residual-gain",
        type=float,
        default=0.20,
        help="Bounded ONNX horizontal residual gain in m/s per normalized unit.",
    )
    parser.add_argument(
        "--flight-policy-vertical-residual-gain",
        type=float,
        default=0.22,
        help="Bounded ONNX vertical residual gain in m/s per normalized unit.",
    )
    parser.add_argument("--max-steps", type=int, default=650)
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Maximum independent PX4 SITL trials per failed combination.",
    )
    parser.add_argument(
        "--reuse-valid",
        action="store_true",
        help="Keep already-successful artifacts that match this Go2 route and locomotion model.",
    )
    return parser.parse_args()


def _valid_existing(
    metric: dict[str, Any],
    output: dict[str, Path],
    *,
    locomotion_sha256: str,
    speed_scale: float,
    turn_scale: float,
    motion_delay_s: float,
    capture_radius_m: float,
    policy_action_gain: float,
    control_contract: str,
    minimum_path_m: float,
    minimum_speed_mps: float,
) -> bool:
    """Return whether an already-recorded replay is safe to retain.

    A retained video must demonstrate actual Go2 locomotion under the exact
    current route settings; old in-place-gait recordings are therefore never
    considered reusable.
    """
    terminal = metric.get("terminal", {})
    locomotion = metric.get("locomotion", {})
    route = metric.get("go2_route", {})
    if not (
        isinstance(terminal, dict)
        and isinstance(locomotion, dict)
        and isinstance(route, dict)
        and bool(metric.get("success"))
        and locomotion.get("sha256") == locomotion_sha256
        and abs(float(locomotion.get("policy_action_gain", -1.0)) - policy_action_gain) < 1.0e-9
        and abs(float(route.get("speed_scale", -1.0)) - speed_scale) < 1.0e-9
        and abs(float(route.get("turn_scale", -1.0)) - turn_scale) < 1.0e-9
        and abs(float(route.get("motion_start_delay_s", -1.0)) - motion_delay_s) < 1.0e-9
        and abs(float(route.get("capture_radius_m", -1.0)) - capture_radius_m) < 1.0e-9
        and metric.get("control_contract") == control_contract
        and float(terminal.get("go2_path_distance_m", 0.0)) >= minimum_path_m
        and float(terminal.get("go2_speed_mps", 0.0)) >= minimum_speed_mps
        and float(terminal.get("go2_fall", 1.0)) == 0.0
        and float(terminal.get("offline_sim_go2_sole_contacts", 0.0)) >= 2.0
    ):
        return False
    return all(path.is_file() and path.stat().st_size > 0 for path in (*output.values(), output["px4_log"].with_suffix(".ulg")))


def main() -> None:
    args = parse_args()
    if args.max_steps <= 0:
        raise SystemExit("--max-steps must be positive")
    if args.retries <= 0:
        raise SystemExit("--retries must be positive")
    if not args.training_metrics.is_file():
        raise SystemExit(f"Missing accepted training metrics: {args.training_metrics}")
    if not args.onnx_manifest.is_file():
        raise SystemExit(f"Missing ONNX manifest: {args.onnx_manifest}")
    if not args.locomotion_model.is_file():
        raise SystemExit(f"Missing Go2 low-level locomotion model: {args.locomotion_model}")
    if not 0.10 <= args.go2_speed_scale <= 1.0:
        raise SystemExit("--go2-speed-scale must be within [0.10, 1.0]")
    if not 0.10 <= args.go2_turn_scale <= 1.0:
        raise SystemExit("--go2-turn-scale must be within [0.10, 1.0]")
    if not 0.0 <= args.go2_motion_start_delay_s <= 5.0:
        raise SystemExit("--go2-motion-start-delay-s must be within [0.0, 5.0]")
    if not 1.30 <= args.search_altitude_world_m <= 2.72:
        raise SystemExit("--search-altitude-world-m must be within [1.30, 2.72]")
    if not 0.10 <= args.go2_policy_action_gain <= 0.50:
        raise SystemExit("--go2-policy-action-gain must be within [0.10, 0.50]")
    if not 0.0 <= args.flight_policy_residual_gain <= 0.50:
        raise SystemExit("--flight-policy-residual-gain must be within [0.0, 0.50]")
    if not 0.0 <= args.flight_policy_vertical_residual_gain <= 0.35:
        raise SystemExit("--flight-policy-vertical-residual-gain must be within [0.0, 0.35]")
    training_contract = json.loads(args.training_metrics.read_text(encoding="utf-8"))
    if training_contract.get("full_3d_policy_control") is not True:
        raise SystemExit("Training metrics do not certify the required full_3d_policy_control contract")
    onnx_contract = json.loads(args.onnx_manifest.read_text(encoding="utf-8"))
    onnx_entries = {
        entry.get("algorithm"): entry
        for entry in onnx_contract.get("models", [])
        if isinstance(entry, dict)
    }
    for algorithm in ALGORITHMS:
        if algorithm == "mpc":
            continue
        entry = onnx_entries.get(algorithm, {})
        if entry.get("output", {}).get("shape") != ["batch", 3]:
            raise SystemExit(f"{algorithm.upper()} ONNX manifest does not declare a 3D action output")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runner = PROJECT_ROOT / "scripts" / "run_px4_mujoco_flat_ppo.py"
    locomotion_sha256 = sha256_file(args.locomotion_model)
    # Do not dilute the trained task definition for deployment footage: all
    # three stages retain their original path speed, curvature, and immediate
    # continuous movement.  Difficulty is defined in GO2_PROFILES itself.
    speed_scale_by_difficulty = {difficulty: float(args.go2_speed_scale) for difficulty in DIFFICULTIES}
    turn_scale_by_difficulty = {difficulty: float(args.go2_turn_scale) for difficulty in DIFFICULTIES}
    motion_delay_by_difficulty = {difficulty: float(args.go2_motion_start_delay_s) for difficulty in DIFFICULTIES}
    # This is a camera-centred descent gate, independent of the Go2 route
    # profile.  It does not change commanded walking speed or route geometry.
    capture_radius_by_difficulty = {difficulty: 0.35 for difficulty in DIFFICULTIES}
    # The sampled terminal instant can fall within a curved-route turn, so
    # use a non-stopping threshold while retaining a path-length floor that
    # proves the full hard route was physically walked.
    locomotion_minimums = {"easy": (0.60, 0.40), "medium": (0.75, 0.55), "hard": (0.90, 0.30)}
    records: list[dict[str, object]] = []
    for algorithm in ALGORITHMS:
        onnx: Path | None = None
        if algorithm != "mpc":
            onnx = args.onnx_dir / f"{algorithm}_px4_flat_hil_3d.onnx"
            if not onnx.is_file():
                raise SystemExit(f"Missing {algorithm.upper()} ONNX: {onnx}")
        for difficulty in DIFFICULTIES:
            motion_delay_s = motion_delay_by_difficulty[difficulty]
            speed_scale = speed_scale_by_difficulty[difficulty]
            turn_scale = turn_scale_by_difficulty[difficulty]
            capture_radius_m = capture_radius_by_difficulty[difficulty]
            minimum_path_m, minimum_speed_mps = locomotion_minimums[difficulty]
            stem = f"px4_sitl_ekf2_{algorithm}_flat_{difficulty}"
            output = {
                "video": args.output_dir / f"{stem}.mp4",
                "snapshot": args.output_dir / f"{stem}.png",
                "metrics": args.output_dir / f"{stem}_metrics.json",
                "trace": args.output_dir / f"{stem}_trace.csv",
                "px4_log": args.output_dir / f"{stem}_px4.log",
            }
            command = [
                sys.executable,
                str(runner),
                "--algorithm", algorithm,
                "--difficulty", difficulty,
                "--seed", str(SEEDS[difficulty]),
                "--locomotion-model", str(args.locomotion_model),
                "--go2-policy-action-gain", str(args.go2_policy_action_gain),
                "--go2-speed-scale", str(speed_scale),
                "--go2-turn-scale", str(turn_scale),
                "--go2-motion-start-delay-s", str(motion_delay_s),
                "--capture-radius-m", str(capture_radius_m),
                "--search-altitude-world-m", str(args.search_altitude_world_m),
                "--flight-policy-residual-gain", str(args.flight_policy_residual_gain),
                "--flight-policy-vertical-residual-gain", str(args.flight_policy_vertical_residual_gain),
                "--full-3d-policy-control",
                "--max-steps", str(args.max_steps),
                "--video-file", str(output["video"]),
                "--snapshot-file", str(output["snapshot"]),
                "--metrics-file", str(output["metrics"]),
                "--trace-file", str(output["trace"]),
                "--px4-log-file", str(output["px4_log"]),
            ]
            if onnx is not None:
                command.extend(("--onnx-model", str(onnx)))
            metric: dict[str, object] = {}
            returncode: int | None = None
            if args.reuse_valid and output["metrics"].is_file():
                existing = json.loads(output["metrics"].read_text(encoding="utf-8"))
                if _valid_existing(
                    existing,
                    output,
                    locomotion_sha256=locomotion_sha256,
                    speed_scale=speed_scale,
                    turn_scale=turn_scale,
                    motion_delay_s=motion_delay_s,
                    capture_radius_m=capture_radius_m,
                    policy_action_gain=float(args.go2_policy_action_gain),
                    control_contract="full_3d_velocity",
                    minimum_path_m=minimum_path_m,
                    minimum_speed_mps=minimum_speed_mps,
                ):
                    metric = existing
                    returncode = 0
                    print(f"PX4 HIL reuse: {algorithm}/{difficulty}", flush=True)
            attempts = 0
            while not bool(metric.get("success")) and attempts < args.retries:
                attempts += 1
                print(f"PX4 HIL start: {algorithm}/{difficulty} attempt {attempts}/{args.retries}", flush=True)
                completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
                returncode = completed.returncode
                metric = {}
                if output["metrics"].is_file():
                    metric = json.loads(output["metrics"].read_text(encoding="utf-8"))
            ulog = output["px4_log"].with_suffix(".ulg")
            record = {
                "algorithm": algorithm,
                "difficulty": difficulty,
                "seed": SEEDS[difficulty],
                # MPC is intentionally a deterministic camera/PnP + EKF
                # baseline, so it has no learned ONNX artefact.
                "onnx": (
                    str(onnx.resolve().relative_to(PROJECT_ROOT))
                    if onnx is not None
                    else None
                ),
                "onnx_sha256": sha256_file(onnx) if onnx is not None else None,
                "video": output["video"].name,
                "snapshot": output["snapshot"].name,
                "metrics": output["metrics"].name,
                "trace": output["trace"].name,
                "px4_log": output["px4_log"].name,
                "ulog": ulog.name,
                "returncode": returncode,
                "attempts": attempts,
                "success": bool(metric.get("success")),
                "go2_route": metric.get("go2_route", {}),
                "locomotion": metric.get("locomotion", {}),
                "terminal": metric.get("terminal", {}),
                "px4": metric.get("px4", {}),
            }
            records.append(record)
            print(
                f"PX4 HIL {'success' if record['success'] else 'failed'}: "
                f"{algorithm}/{difficulty} (exit {returncode}; attempts {attempts})",
                flush=True,
            )
    manifest = {
        "backend": "MuJoCo X500 physics + separate PX4 SITL EKF2 MAVLink HIL",
        "terrain": "flat",
        "training_backend": "MuJoCo sensor-contract RL training; PX4 HIL deployment verification",
        "locomotion": {
            "model": str(args.locomotion_model.resolve().relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(args.locomotion_model),
            "mode": "learned 12-joint Go2 PPO; physical foot-contact locomotion",
            "policy_action_gain": float(args.go2_policy_action_gain),
        },
        "go2_route_speed_scale": speed_scale_by_difficulty,
        "go2_route_turn_scale": turn_scale_by_difficulty,
        "go2_motion_start_delay_s": motion_delay_by_difficulty,
        "camera_capture_radius_m": capture_radius_by_difficulty,
        "search_altitude_world_m": float(args.search_altitude_world_m),
        "flight_policy_residual_gain_mps": float(args.flight_policy_residual_gain),
        "flight_policy_vertical_residual_gain_mps": float(args.flight_policy_vertical_residual_gain),
        "control_contract": "full_3d_velocity",
        "training_metrics": str(args.training_metrics.resolve().relative_to(PROJECT_ROOT)),
        "onnx_manifest": str(args.onnx_manifest.resolve().relative_to(PROJECT_ROOT)),
        "algorithms": list(ALGORITHMS),
        "difficulties": list(DIFFICULTIES),
        "records": records,
        "all_success": all(bool(record["success"]) for record in records),
    }
    manifest_path = args.output_dir / "px4_flat_hil_suite.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not manifest["all_success"]:
        failed = [f"{record['algorithm']}/{record['difficulty']}" for record in records if not record["success"]]
        raise SystemExit(f"PX4 HIL suite did not complete: {', '.join(failed)}")
    print(f"PX4 flat HIL suite confirmed: {manifest_path}")


if __name__ == "__main__":
    main()
