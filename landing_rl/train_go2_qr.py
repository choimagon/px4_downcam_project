"""Train PPO, DDPG and SAC for landing on the moving Unitree Go2 QR deck."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import shutil
import uuid
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from .go2_qr_environment import (
    DRONE_OBSERVATION_NAMES,
    GO2_PROFILES,
    IMU_IMPACT_MAX_VISUAL_HEIGHT_M,
    IMU_SETTLE_THRUST_FRACTION,
    LANDING_POLICY_RESIDUAL_SPEED_MPS,
    LANDING_POLICY_TRAINING_RESIDUAL_SPEED_MPS,
    QR_INK_RENDER_CLEARANCE_M,
    SUCCESS_MAX_RELATIVE_HEIGHT_M,
    X500_NOMINAL_TOUCHDOWN_RELATIVE_HEIGHT_M,
    X500_SKID_CENTER_BODY_Z_M,
    X500_SKID_HALF_SIZE_M,
    X500_SKID_LATERAL_OFFSET_M,
    X500_VISUAL_SKID_BOTTOM_BODY_Z_M,
    Go2BackQrLandingEnv,
)
from .train import ALGORITHMS, make_model


REQUIRED_ALGORITHMS = ("ppo", "ddpg", "sac")
REQUIRED_EVALUATION_DIFFICULTIES = ("easy", "medium", "hard")
MAX_OFFLINE_SIM_PENETRATION_M = 0.002


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    """Replace a JSON manifest only after its complete contents are durable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def promote_model_set_atomically(
    *,
    candidates: dict[str, Path],
    canonicals: dict[str, Path],
    best_algorithm: str,
    best_path: Path,
    manifest_source: Path,
    manifest_path: Path,
) -> None:
    """Transactionally replace three policies, best alias, and accepted manifest.

    All source archives are copied to same-filesystem pending files before any
    canonical is touched.  A failed replacement restores every old target, so
    one algorithm cannot be promoted independently of the other two.
    """
    if tuple(candidates) != REQUIRED_ALGORITHMS or tuple(canonicals) != REQUIRED_ALGORITHMS:
        raise ValueError(f"model promotion requires exactly: {', '.join(REQUIRED_ALGORITHMS)}")
    if best_algorithm not in candidates:
        raise ValueError(f"unknown best algorithm: {best_algorithm}")
    for source in candidates.values():
        if not source.is_file():
            raise FileNotFoundError(source)

    replacements = [(candidates[name], canonicals[name]) for name in REQUIRED_ALGORITHMS]
    replacements.append((candidates[best_algorithm], best_path))
    replacements.append((manifest_source, manifest_path))
    token = uuid.uuid4().hex
    pending: dict[Path, Path] = {}
    backups: dict[Path, Path] = {}
    installed: list[Path] = []
    succeeded = False
    try:
        for source, target in replacements:
            target.parent.mkdir(parents=True, exist_ok=True)
            staged = target.with_name(f".{target.name}.{token}.pending")
            shutil.copy2(source, staged)
            pending[target] = staged
        for _, target in replacements:
            if target.exists():
                backup = target.with_name(f".{target.name}.{token}.backup")
                target.replace(backup)
                backups[target] = backup
        for _, target in replacements:
            pending[target].replace(target)
            installed.append(target)
        succeeded = True
    except BaseException as promotion_error:
        rollback_errors: list[str] = []
        for target in reversed(installed):
            try:
                target.unlink(missing_ok=True)
            except OSError as error:
                rollback_errors.append(f"remove {target}: {error}")
        for target, backup in reversed(tuple(backups.items())):
            try:
                if target.exists() or target.is_symlink():
                    target.unlink()
                backup.replace(target)
            except OSError as error:
                rollback_errors.append(f"restore {target}: {error}")
        if rollback_errors:
            raise RuntimeError(
                f"model-set promotion failed ({promotion_error}) and rollback was incomplete: "
                + "; ".join(rollback_errors)
            ) from promotion_error
        raise
    finally:
        for staged in pending.values():
            staged.unlink(missing_ok=True)
        if succeeded:
            for backup in backups.values():
                backup.unlink(missing_ok=True)


def evaluate_landing(
    model: Any, *, seed: int, episodes: int, difficulty: str, locomotion_model: Path
) -> dict[str, float]:
    """Evaluate six primary metrics plus explicitly offline MuJoCo diagnostics."""
    env = Go2BackQrLandingEnv(seed=seed, difficulty=difficulty, locomotion_model=locomotion_model)
    rewards: list[float] = []
    terminal_errors: list[float] = []
    terminal_steps: list[float] = []
    offline_sim_contact_counts: list[float] = []
    offline_sim_contact_forces: list[float] = []
    offline_sim_penetrations: list[float] = []
    offline_sim_path_distances: list[float] = []
    offline_sim_pad_speeds: list[float] = []
    offline_sim_go2_speeds: list[float] = []
    offline_sim_go2_stance_slips: list[float] = []
    offline_sim_go2_base_heights: list[float] = []
    offline_sim_go2_tilts: list[float] = []
    offline_sim_go2_root_wrench_maxima: list[float] = []
    falls = 0
    successes = 0
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        total_reward = 0.0
        last_info: dict[str, float] = {}
        while True:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, last_info = env.step(np.asarray(action, dtype=np.float32))
            total_reward += reward
            # These quantities are read only by the evaluator.  They are not
            # members of DRONE_OBSERVATION_NAMES and never reach model.predict.
            base_up = float(np.clip(env.data.xmat[env.base_id, 8], -1.0, 1.0))
            offline_sim_go2_speeds.append(float(np.linalg.norm(env.data.qvel[:2])))
            offline_sim_go2_stance_slips.append(float(env._go2_stance_slip_mps))
            offline_sim_go2_base_heights.append(float(env.base_position[2]))
            offline_sim_go2_tilts.append(float(np.degrees(np.arccos(base_up))))
            offline_sim_go2_root_wrench_maxima.append(float(np.max(np.abs(env.data.xfrc_applied[env.base_id]))))
            if terminated or truncated:
                break
        successes += int(last_info.get("success", 0.0) > 0.5)
        falls += int(last_info.get("go2_fall", 0.0) > 0.5)
        rewards.append(total_reward)
        terminal_errors.append(last_info.get("horizontal_error_m", float("inf")))
        terminal_steps.append(last_info.get("episode_steps", float("inf")))
        offline_sim_contact_counts.append(float(last_info["offline_sim_landing_skid_contacts"]))
        offline_sim_contact_forces.append(float(last_info["offline_sim_landing_normal_force_n"]))
        offline_sim_penetrations.append(float(last_info["offline_sim_max_contact_penetration_m"]))
        offline_sim_path_distances.append(float(env._path_length))
        offline_sim_pad_speeds.append(float(np.linalg.norm(env._pad_velocity)))
    mean_episode_steps = float(np.mean(terminal_steps))
    mean_episode_duration_s = env.dt * mean_episode_steps
    env.close()
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "success_rate": successes / episodes,
        "mean_terminal_error_m": float(np.mean(terminal_errors)),
        "mean_episode_duration_s": mean_episode_duration_s,
        "mean_episode_steps": mean_episode_steps,
        "offline_sim_go2_fall_rate": falls / episodes,
        "offline_sim_mean_landing_skid_contacts": float(np.mean(offline_sim_contact_counts)),
        "offline_sim_mean_landing_normal_force_n": float(np.mean(offline_sim_contact_forces)),
        "offline_sim_mean_max_penetration_m": float(np.mean(offline_sim_penetrations)),
        "offline_sim_worst_max_penetration_m": float(np.max(offline_sim_penetrations)),
        "offline_sim_mean_go2_path_distance_m": float(np.mean(offline_sim_path_distances)),
        "offline_sim_mean_pad_speed_mps": float(np.mean(offline_sim_pad_speeds)),
        "offline_sim_mean_go2_speed_mps": float(np.mean(offline_sim_go2_speeds)),
        "offline_sim_mean_go2_stance_slip_mps": float(np.mean(offline_sim_go2_stance_slips)),
        "offline_sim_mean_go2_base_height_m": float(np.mean(offline_sim_go2_base_heights)),
        "offline_sim_mean_go2_tilt_deg": float(np.mean(offline_sim_go2_tilts)),
        "offline_sim_go2_root_wrench_max_abs": float(np.max(offline_sim_go2_root_wrench_maxima)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithms", default="ppo,ddpg,sac")
    parser.add_argument("--timesteps", type=int, default=16_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/rl_training"))
    parser.add_argument("--model-suffix", default="go2_back_qr")
    parser.add_argument("--metrics-file", default="go2_back_qr_training_metrics.json")
    parser.add_argument("--evaluation-difficulties", default="easy,medium,hard")
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Re-evaluate saved policies without changing their learned weights (useful when only a held-out profile changes)",
    )
    parser.add_argument(
        "--locomotion-model",
        type=Path,
        default=Path("models/go2_legged_loco_ppo.zip"),
        help="MuJoCo PPO residual policy trained from the legged-loco Go2 task contract",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = [name.strip().lower() for name in args.algorithms.split(",") if name.strip()]
    if selected != list(REQUIRED_ALGORITHMS):
        raise SystemExit(f"--algorithms must be exactly: {','.join(REQUIRED_ALGORITHMS)}")
    difficulties = [value.strip().lower() for value in args.evaluation_difficulties.split(",") if value.strip()]
    if difficulties != list(REQUIRED_EVALUATION_DIFFICULTIES):
        raise SystemExit(
            "--evaluation-difficulties must be exactly: "
            f"{','.join(REQUIRED_EVALUATION_DIFFICULTIES)}"
        )
    if args.eval_episodes <= 0:
        raise SystemExit("--eval-episodes must be positive")
    if not args.evaluate_only and args.timesteps <= 0:
        raise SystemExit("--timesteps must be positive")
    if not args.locomotion_model.is_file():
        raise SystemExit(f"--locomotion-model does not exist: {args.locomotion_model}")
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    # The suite takes the same advisory lock while it snapshots/exports and
    # records all nine demonstrations.  Holding it for this whole process
    # prevents a concurrent training run from changing model generations
    # between preflight, ONNX export and publication.
    model_set_lock = (args.models_dir / ".go2_back_qr_model_set.lock").open("a+")
    fcntl.flock(model_set_lock.fileno(), fcntl.LOCK_EX)
    set_random_seed(args.seed)
    previous_metrics: dict[str, Any] = {}
    if args.evaluate_only:
        metrics_path = args.artifacts_dir / args.metrics_file
        if not metrics_path.is_file():
            raise SystemExit(f"--evaluate-only needs an existing metrics file: {metrics_path}")
        previous_manifest = json.loads(metrics_path.read_text(encoding="utf-8"))
        previous_metrics = previous_manifest.get("metrics", {})
        previous_promotion = previous_manifest.get("promotion", {})
        previous_hashes = previous_promotion.get("canonical_model_sha256", {})
        expected_archives = {
            name: args.models_dir / f"{name}_{args.model_suffix}.zip"
            for name in REQUIRED_ALGORITHMS
        }
        provenance_ok = (
            previous_manifest.get("algorithms") == list(REQUIRED_ALGORITHMS)
            and tuple(previous_manifest.get("observation_names", ())) == DRONE_OBSERVATION_NAMES
            and previous_manifest.get("evaluation_difficulties") == list(REQUIRED_EVALUATION_DIFFICULTIES)
            and previous_promotion.get("passed") is True
            and previous_promotion.get("atomic_model_set") is True
            and previous_promotion.get("status") in {"promoted_atomically", "validated_existing"}
            and isinstance(previous_hashes, dict)
            and set(previous_hashes) == set(REQUIRED_ALGORITHMS)
            and all(
                archive.is_file() and previous_hashes.get(name) == sha256_file(archive)
                for name, archive in expected_archives.items()
            )
            and previous_manifest.get("locomotion_model_sha256") == sha256_file(args.locomotion_model)
        )
        if not provenance_ok:
            raise SystemExit(
                "--evaluate-only refused: the existing model set lacks a matching "
                "sensor-only atomic-promotion provenance record"
            )
    results: dict[str, dict[str, Any]] = {}
    best_name, best_score = "", float("-inf")
    rejected_algorithms: list[str] = []
    for offset, name in enumerate(selected):
        seed = args.seed + offset
        model_path = args.models_dir / f"{name}_{args.model_suffix}"
        if args.evaluate_only:
            if name not in previous_metrics:
                raise SystemExit(f"--evaluate-only has no saved metrics for {name}")
            archive = model_path.with_suffix(".zip")
            if not archive.is_file():
                raise SystemExit(f"--evaluate-only is missing model: {archive}")
            model = ALGORITHMS[name].load(archive, device="cpu")
            training = evaluate_landing(
                model,
                seed=seed + 10_000,
                episodes=args.eval_episodes,
                difficulty="train",
                locomotion_model=args.locomotion_model,
            )
            held_out = {}
        else:
            environment = Monitor(
                Go2BackQrLandingEnv(
                    seed=seed,
                    difficulty="train",
                    locomotion_model=args.locomotion_model,
                    policy_residual_speed_mps=LANDING_POLICY_TRAINING_RESIDUAL_SPEED_MPS,
                )
            )
            model = make_model(name, environment, seed)
            model.learn(total_timesteps=args.timesteps, progress_bar=False)
            candidate_path = args.models_dir / f"{name}_{args.model_suffix}_candidate"
            model.save(candidate_path)
            environment.close()
            # Evaluate the exact bytes that can be promoted/exported.  This
            # prevents an in-memory learner state from passing while the
            # serialized deployment archive behaves differently after load.
            model = ALGORITHMS[name].load(candidate_path.with_suffix(".zip"), device="cpu")
            training = evaluate_landing(
                model, seed=seed + 10_000, episodes=args.eval_episodes, difficulty="train", locomotion_model=args.locomotion_model
            )
            held_out = {}
        held_out.update({
            difficulty: evaluate_landing(
                model,
                seed=seed + 20_000 + index * 1_000,
                episodes=args.eval_episodes,
                difficulty=difficulty,
                locomotion_model=args.locomotion_model,
            )
            for index, difficulty in enumerate(difficulties)
        })
        evaluated_profiles = {"training": training, **held_out}
        acceptance = {
            "all_profiles_success_rate_ge_0_95": all(
                values["success_rate"] >= 0.95 for values in evaluated_profiles.values()
            ),
            "all_profiles_terminal_error_le_0_055m": all(
                values["mean_terminal_error_m"] <= 0.055 for values in evaluated_profiles.values()
            ),
            "all_profiles_go2_fall_rate_zero": all(
                values["offline_sim_go2_fall_rate"] == 0.0 for values in evaluated_profiles.values()
            ),
            "all_profiles_root_wrench_zero": all(
                values["offline_sim_go2_root_wrench_max_abs"] == 0.0 for values in evaluated_profiles.values()
            ),
            "all_profiles_worst_penetration_le_2mm": all(
                values["offline_sim_worst_max_penetration_m"] <= MAX_OFFLINE_SIM_PENETRATION_M
                for values in evaluated_profiles.values()
            ),
        }
        acceptance["passed"] = all(acceptance.values())
        results[name] = {"training": training, "held_out": held_out, "acceptance": acceptance}
        if not acceptance["passed"]:
            rejected_algorithms.append(name)
        medium = held_out.get("medium", next(iter(held_out.values())))
        score = (
            medium["success_rate"] * 100.0
            - 100.0 * medium["offline_sim_go2_fall_rate"]
            - medium["mean_terminal_error_m"]
            if acceptance["passed"]
            else float("-inf")
        )
        if score > best_score:
            best_name, best_score = name, score
        if not args.evaluate_only:
            environment.close()
        print(f"{name}: {json.dumps(results[name], sort_keys=True)}", flush=True)

    all_models_passed = (
        tuple(results) == REQUIRED_ALGORITHMS
        and not rejected_algorithms
        and all(results[name]["acceptance"]["passed"] for name in REQUIRED_ALGORITHMS)
    )
    promotion_status = (
        "validated_existing"
        if args.evaluate_only and all_models_passed
        else "promoted_atomically"
        if all_models_passed
        else "rejected_candidates_preserved"
    )
    candidate_archives = {
        name: args.models_dir / f"{name}_{args.model_suffix}_candidate.zip"
        for name in REQUIRED_ALGORITHMS
    }
    canonical_archives = {
        name: args.models_dir / f"{name}_{args.model_suffix}.zip"
        for name in REQUIRED_ALGORITHMS
    }
    candidate_hashes = (
        {name: sha256_file(path) for name, path in candidate_archives.items()}
        if not args.evaluate_only
        else {}
    )
    canonical_hashes = (
        {
            name: (
                sha256_file(canonical_archives[name])
                if args.evaluate_only
                else candidate_hashes[name]
            )
            for name in REQUIRED_ALGORITHMS
        }
        if all_models_passed
        else {}
    )
    manifest = {
        "backend": "MuJoCo",
        "mujoco_version": mujoco.__version__,
        "seed": args.seed,
        "timesteps_per_algorithm": args.timesteps,
        "eval_episodes_per_difficulty": args.eval_episodes,
        "algorithms": selected,
        "best_algorithm": best_name if all_models_passed else "",
        "metrics": results,
        "training_difficulty": "train",
        "evaluation_difficulties": difficulties,
        "max_offline_sim_penetration_m": MAX_OFFLINE_SIM_PENETRATION_M,
        "primary_evaluation_metrics": [
            "mean_reward",
            "std_reward",
            "success_rate",
            "mean_terminal_error_m",
            "mean_episode_duration_s",
            "mean_episode_steps",
        ],
        "offline_sim_diagnostics": [
            "offline_sim_go2_fall_rate",
            "offline_sim_mean_landing_skid_contacts",
            "offline_sim_mean_landing_normal_force_n",
            "offline_sim_mean_max_penetration_m",
            "offline_sim_worst_max_penetration_m",
            "offline_sim_mean_go2_path_distance_m",
            "offline_sim_mean_pad_speed_mps",
            "offline_sim_mean_go2_speed_mps",
            "offline_sim_mean_go2_stance_slip_mps",
            "offline_sim_mean_go2_base_height_m",
            "offline_sim_mean_go2_tilt_deg",
            "offline_sim_go2_root_wrench_max_abs",
        ],
        "go2_model_source": "unitreerobotics/unitree_mujoco @ 4134cb5 (official Go2 MJCF + meshes)",
        "locomotion_reference": "MuJoCo PPO residual retrained from yang-zj1026/legged-loco Go2 task contract; a_deploy=0.50*clip(a_PPO,-1,1), q_target=q_ref+0.18*a_deploy, 58% duty diagonal trot [0,0.5,0.5,0], 60/2 tracking PD, identically zero six-axis root wrench, foot-slip and body-height rewards",
        "locomotion_model": str(args.locomotion_model),
        "locomotion_model_sha256": sha256_file(args.locomotion_model),
        "observation": "[qr_center_u, qr_center_v, qr_pnp_depth, qr_detected, qr_center_rate_u, qr_center_rate_v, drone_vertical_velocity]",
        "observation_names": DRONE_OBSERVATION_NAMES,
        "observation_formula": "[u_qr, v_qr, min(1,z_pnp/8), detected, clip(du_qr/dt/3), clip(dv_qr/dt/3), clip(vz_est/3)] in [-1,1]^7",
        "observation_source": "30 Hz stock X500 downward-camera QR detector/solvePnP and consecutive QR-center rate plus 50 Hz PX4 estimator vertical velocity only; no landing-gear sensor, Go2, pad, base, route or simulator target state",
        "attitude_control_source": "30 Hz noisy sample-and-held camera-relative QR solvePnP rotation, reconstructed in world coordinates using only the calibrated camera-to-body extrinsic and X500 onboard attitude estimate; exact marker/world rotation is not exposed",
        "camera_pnp_rotation_noise": "seeded Gaussian rotation-vector noise: sigma_deg=0.15+0.03*depth_m, clipped per axis at 3 sigma and held at 30 Hz",
        "sensor_rates_hz": {"downward_camera": 30, "px4_estimator": 50},
        "x500_landing_geometry": {
            "source": "two continuous visible stock PX4 Gazebo X500 landing-sole collision geoms aligned to the imported stock visual frame mesh and the physical QR board top",
            "visual_skid_sole_body_z_m": X500_VISUAL_SKID_BOTTOM_BODY_Z_M,
            "rail_center_body_z_m": X500_SKID_CENTER_BODY_Z_M,
            "rail_half_size_m": list(X500_SKID_HALF_SIZE_M),
            "rail_sole_body_z_m": (
                X500_SKID_CENTER_BODY_Z_M - X500_SKID_HALF_SIZE_M[2]
            ),
            "visual_contact_plane_error_m": abs(
                X500_SKID_CENTER_BODY_Z_M
                - X500_SKID_HALF_SIZE_M[2]
                - X500_VISUAL_SKID_BOTTOM_BODY_Z_M
            ),
            "rail_center_xy_m": [[0.0, X500_SKID_LATERAL_OFFSET_M], [0.0, -X500_SKID_LATERAL_OFFSET_M]],
            "rail_condim": 3,
            "rails_rendered": True,
            "qr_ink_render_clearance_m": QR_INK_RENDER_CLEARANCE_M,
            "camera_placeholder_rendered": False,
            "nominal_touchdown_relative_height_m": X500_NOMINAL_TOUCHDOWN_RELATIVE_HEIGHT_M,
            "touchdown_down_camera_depth_m": (
                X500_NOMINAL_TOUCHDOWN_RELATIVE_HEIGHT_M - 0.065
            ),
            "success_max_relative_height_m": SUCCESS_MAX_RELATIVE_HEIGHT_M,
        },
        "go2_action_contract": "a_deploy=0.50*clip(a_PPO,-1,1); q_target=q_ref+0.18*a_deploy; applied six-axis root wrench=0",
        "action": "[lateral_x, lateral_y] proposal projected onto the inward camera QR-error direction only; training exploration envelope 0.002 m/s, held-out evaluation/deployment envelope 0.001 m/s, tapered below 1.20 m and zero inside 0.45 m; no target-state feed-forward",
        "training_policy_residual_speed_mps": LANDING_POLICY_TRAINING_RESIDUAL_SPEED_MPS,
        "deployment_policy_residual_speed_mps": LANDING_POLICY_RESIDUAL_SPEED_MPS,
        "search": "while QR is absent, use only the declared forward-corridor mission waypoint, elapsed time and own 50 Hz PX4 position/altitude estimate for a lateral sweep; begin target tracking only after a 30 Hz camera detection",
        "imu_landing_controller": f"body-Z accelerometer minus known commanded body-Z specific force; impact gate 4.0 m/s^2 below {IMU_IMPACT_MAX_VISUAL_HEIGHT_M:.3f} m visual height and |vz_est|<=0.45 m/s; cut collective to {IMU_SETTLE_THRUST_FRACTION:.2f} of hover for the 0.35 s settle window instead of bouncing the leading skid, then climb at 0.45 m/s for visual reacquisition only if not settled; last-target memory is used only for a genuine detector dropout/brief occlusion because the down camera remains about {X500_NOMINAL_TOUCHDOWN_RELATIVE_HEIGHT_M - 0.065:.3f} m above the marker at stock-skid touchdown; no landing-leg/contact input",
        "start_distribution": "X500 begins uniformly at a random 2–7 m annulus position around the QR deck fixed to Go2 base_link",
        "qr_mount": "0.22 kg rigid base_link child; 23 cm QR print overlays the visible 36 cm physical landing plate. The colliding board top is the QR floor; a 3 micrometre ink-only render clearance prevents coplanar camera z-fighting, with no hidden collision cap or visible gap",
        "contact_calibration": "offline_sim-only diagnostic: the two continuous visible stock PX4 Gazebo X500 landing-sole geoms are each 0.25x0.015x0.015 m at x=0,y=+/-0.132 m; their sole z=-0.22759951 m is aligned to the imported rendered skid and contacts the physical QR board top beneath a 3 micrometre ink-only render layer; success requires both soles on the visible deck; the decorative camera housing/lens are hidden while the actual down_camera remains active; never a policy sensor; condim 3, friction 0.95/0.015/0.001, solref 0.008 1, solimp 0.96 0.99 0.001",
        "physics": "official 12-DoF Unitree Go2 MJCF + X500 free body; 5 ms MuJoCo RK4 integration, 100 ms policy control",
        "profiles": GO2_PROFILES,
        "promotion": {
            "passed": all_models_passed,
            "status": promotion_status,
            "atomic_model_set": True,
            "required_algorithms": list(REQUIRED_ALGORITHMS),
            "required_profiles": ["train", *REQUIRED_EVALUATION_DIFFICULTIES],
            "canonical_model_sha256": canonical_hashes,
            "candidate_model_sha256": candidate_hashes,
        },
    }
    metrics_path = args.artifacts_dir / args.metrics_file
    if all_models_passed and not args.evaluate_only:
        staged_manifest = metrics_path.with_name(f".{metrics_path.name}.{uuid.uuid4().hex}.accepted")
        try:
            write_json_atomic(staged_manifest, manifest)
            promote_model_set_atomically(
                candidates=candidate_archives,
                canonicals=canonical_archives,
                best_algorithm=best_name,
                best_path=args.models_dir / f"best_{args.model_suffix}.zip",
                manifest_source=staged_manifest,
                manifest_path=metrics_path,
            )
        finally:
            staged_manifest.unlink(missing_ok=True)
    elif all_models_passed:
        write_json_atomic(metrics_path, manifest)
    else:
        # Preserve the last accepted manifest alongside the preserved
        # canonical archives.  Failed candidate evidence is still retained in
        # a separate deterministic report for diagnosis and retraining.
        rejected_metrics_path = metrics_path.with_name(
            f"{metrics_path.stem}_rejected{metrics_path.suffix}"
        )
        write_json_atomic(rejected_metrics_path, manifest)
        print(f"rejected evaluation report: {rejected_metrics_path}", flush=True)
    if all_models_passed:
        print(f"best policy: {best_name} -> {args.models_dir / f'best_{args.model_suffix}.zip'}", flush=True)
    else:
        print("best policy: none (the previous canonical model set was preserved)", flush=True)
    if not all_models_passed:
        raise SystemExit(
            "Rejected landing model set; all three canonical files were preserved. "
            f"Failed algorithms: {', '.join(rejected_algorithms)}"
        )


if __name__ == "__main__":
    main()
