"""Train a MuJoCo Go2 low-level PPO controller from legged-loco's task spec."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from .go2_legged_loco_environment import (
    DEPLOYMENT_POLICY_ACTION_GAIN,
    TERRAIN_RESIDUAL_ACTION_LIMIT,
    TERRAIN_RESIDUAL_POLICY_GAIN,
    Go2LeggedLocoEnv,
)
from .go2_terrain import TERRAIN_TASKS, terrain_metadata


def make_policy(env: Monitor, *, seed: int) -> PPO:
    """Match the upstream Go2 PPO widths and optimization constants."""
    return PPO(
        "MlpPolicy",
        env,
        learning_rate=1.0e-4,
        n_steps=960,
        batch_size=240,
        n_epochs=3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.10,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=True,
        sde_sample_freq=4,
        target_kl=0.012,
        policy_kwargs={
            "net_arch": [512, 256, 128],
            "activation_fn": torch.nn.ELU,
            "log_std_init": -3.0,
            "squash_output": True,
        },
        seed=seed,
        verbose=0,
    )


def evaluate(
    model: PPO | None,
    *,
    seed: int,
    episodes: int,
    domain_randomization: bool = False,
    terrain_task: str = "flat",
    rough_level: int | None = None,
) -> dict[str, float]:
    env = Go2LeggedLocoEnv(
        seed=seed,
        history_length=9,
        domain_randomization=domain_randomization,
        sensor_noise=True,
        terrain_task=terrain_task,
        rough_level=rough_level,
    )
    rewards: list[float] = []
    velocity_errors: list[float] = []
    yaw_errors: list[float] = []
    base_ups: list[float] = []
    paths: list[float] = []
    stance_slips: list[float] = []
    base_heights: list[float] = []
    gait_matches: list[float] = []
    assist_forces: list[float] = []
    all_velocity_errors: list[float] = []
    all_yaw_errors: list[float] = []
    all_heights: list[float] = []
    all_clearances: list[float] = []
    action_saturation: list[float] = []
    raw_policy_action_saturation: list[float] = []
    torque_saturation: list[float] = []
    base_tilts: list[float] = []
    heading_errors: list[float] = []
    lateral_offsets: list[float] = []
    root_wrench_maxima: list[float] = []
    world_x_progresses: list[float] = []
    terminal_lateral_offsets: list[float] = []
    falls = 0
    # Terrain environments scale raw policy actions once internally, exactly
    # like the 5-ms landing-scene controller.  Do not apply a second 25%
    # factor during evaluation.
    deployment_gain = 1.0 if terrain_task != "flat" else DEPLOYMENT_POLICY_ACTION_GAIN
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        start_x = float(env.data.qpos[0])
        start_y = float(env.data.qpos[1])
        total = 0.0
        episode_slips: list[float] = []
        episode_heights: list[float] = []
        episode_matches: list[float] = []
        episode_assists: list[float] = []
        last_info: dict[str, float] = {}
        while True:
            if model is None:
                action = np.zeros(12, dtype=np.float32)
                raw_action = action
            else:
                raw_action, _ = model.predict(observation, deterministic=True)
                raw_action = np.clip(np.asarray(raw_action, dtype=np.float32), -1.0, 1.0)
                action = deployment_gain * raw_action
            observation, reward, terminated, truncated, last_info = env.step(np.asarray(action, dtype=np.float32))
            total += reward
            episode_slips.append(last_info["stance_foot_slip_mps"])
            episode_heights.append(last_info["base_height_m"])
            episode_matches.append(last_info["gait_contact_match"])
            episode_assists.append(last_info["assist_force_n"])
            all_velocity_errors.append(last_info["velocity_error_mps"])
            all_yaw_errors.append(last_info["yaw_rate_error_radps"])
            all_heights.append(last_info["base_height_m"])
            all_clearances.append(last_info["base_height_m"] - last_info["terrain_ground_height_m"])
            applied_action = (
                TERRAIN_RESIDUAL_POLICY_GAIN * np.clip(action, -TERRAIN_RESIDUAL_ACTION_LIMIT, TERRAIN_RESIDUAL_ACTION_LIMIT)
                if terrain_task != "flat" else action
            )
            action_saturation.append(float(np.mean(np.abs(applied_action) > 0.95)))
            raw_policy_action_saturation.append(float(np.mean(np.abs(raw_action) > 0.95)))
            torque_saturation.append(last_info["torque_saturation_fraction"])
            base_tilts.append(last_info["base_tilt_deg"])
            heading_errors.append(last_info["course_heading_error_deg"])
            lateral_offsets.append(abs(last_info["world_y_m"] - start_y))
            root_wrench_maxima.append(last_info["root_wrench_max_abs"])
            if terminated or truncated:
                break
        rewards.append(total)
        velocity_errors.append(last_info["velocity_error_mps"])
        yaw_errors.append(last_info["yaw_rate_error_radps"])
        base_ups.append(last_info["base_up"])
        paths.append(last_info["path_distance_m"])
        stance_slips.append(float(np.mean(episode_slips)))
        base_heights.append(float(np.mean(episode_heights)))
        gait_matches.append(float(np.mean(episode_matches)))
        assist_forces.append(float(np.mean(episode_assists)))
        world_x_progresses.append(last_info["world_x_m"] - start_x)
        terminal_lateral_offsets.append(abs(last_info["world_y_m"] - start_y))
        falls += int(last_info["fall"] > 0.5)
    env.close()
    return {
        "mean_episode_reward": float(np.mean(rewards)),
        "std_episode_reward": float(np.std(rewards)),
        "mean_terminal_velocity_error_mps": float(np.mean(velocity_errors)),
        "mean_terminal_yaw_rate_error_radps": float(np.mean(yaw_errors)),
        "mean_terminal_base_up": float(np.mean(base_ups)),
        "mean_path_distance_m": float(np.mean(paths)),
        "mean_stance_foot_slip_mps": float(np.mean(stance_slips)),
        "mean_base_height_m": float(np.mean(base_heights)),
        "mean_gait_contact_match": float(np.mean(gait_matches)),
        "mean_assist_force_n": float(np.mean(assist_forces)),
        "time_series_velocity_rmse_mps": float(np.sqrt(np.mean(np.square(all_velocity_errors)))),
        "time_series_yaw_rate_rmse_radps": float(np.sqrt(np.mean(np.square(all_yaw_errors)))),
        "base_height_p05_m": float(np.quantile(all_heights, 0.05)),
        "base_height_p95_m": float(np.quantile(all_heights, 0.95)),
        "base_clearance_p05_m": float(np.quantile(all_clearances, 0.05)),
        "base_clearance_p95_m": float(np.quantile(all_clearances, 0.95)),
        "base_tilt_p95_deg": float(np.quantile(base_tilts, 0.95)),
        "course_heading_error_p95_deg": float(np.quantile(heading_errors, 0.95)),
        "course_lateral_offset_p95_m": float(np.quantile(lateral_offsets, 0.95)),
        "mean_world_x_progress_m": float(np.mean(world_x_progresses)),
        "world_x_progress_p05_m": float(np.quantile(world_x_progresses, 0.05)),
        "mean_terminal_lateral_offset_m": float(np.mean(terminal_lateral_offsets)),
        "action_saturation_fraction": float(np.mean(action_saturation)),
        "raw_policy_action_saturation_fraction": float(np.mean(raw_policy_action_saturation)),
        "torque_saturation_fraction": float(np.mean(torque_saturation)),
        "root_wrench_max_abs": float(np.max(root_wrench_maxima)),
        "fall_rate": falls / episodes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=40_000)
    parser.add_argument(
        "--learning-rate", type=float, default=None,
        help="Override PPO learning rate; useful for conservative terrain fine-tuning from an accepted gait.",
    )
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--robust-eval-episodes", type=int, default=30)
    parser.add_argument("--robust-eval-seed", type=int, default=20290831)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--model", type=Path, default=Path("models/go2_legged_loco_ppo"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/rl_training"))
    parser.add_argument(
        "--metrics-file", type=Path, default=None,
        help="Optional manifest path. Defaults to <artifacts-dir>/go2_legged_loco_metrics.json.",
    )
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--terrain", choices=TERRAIN_TASKS, default="flat")
    parser.add_argument(
        "--rough-level", type=int, choices=(1, 2, 3), default=None,
        help="Fix a rough terrain level. Omit for the 1/2/3 curriculum during training/evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.model.parent.mkdir(parents=True, exist_ok=True)
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    set_random_seed(args.seed)
    env = Monitor(
        Go2LeggedLocoEnv(
            seed=args.seed,
            history_length=9,
            terrain_task=args.terrain,
            rough_level=args.rough_level,
        )
    )
    if args.resume_from is None:
        model = make_policy(env, seed=args.seed)
        resumed_from = None
    else:
        model = PPO.load(args.resume_from, env=env, device="cpu")
        resumed_from = str(args.resume_from)
        if args.learning_rate is not None:
            if args.learning_rate <= 0.0:
                raise ValueError("--learning-rate must be positive")
            model.learning_rate = float(args.learning_rate)
            model.lr_schedule = lambda _progress_remaining: float(args.learning_rate)
    model.learn(
        total_timesteps=args.timesteps,
        progress_bar=False,
        reset_num_timesteps=args.resume_from is None,
    )
    canonical_model = args.model if args.model.suffix == ".zip" else args.model.with_suffix(".zip")
    candidate_model = canonical_model.with_name(canonical_model.stem + "_candidate.zip")
    model.save(candidate_model)
    metrics = evaluate(
        model, seed=args.seed + 10_000, episodes=args.eval_episodes,
        terrain_task=args.terrain, rough_level=args.rough_level,
    )
    zero_residual_metrics = evaluate(
        None, seed=args.seed + 10_000, episodes=args.eval_episodes,
        terrain_task=args.terrain, rough_level=args.rough_level,
    )
    # The certificate must survive unseen but physically plausible actuator,
    # friction, gain and sensor perturbations.  A deterministic replay cannot
    # certify uphill or rough-terrain walking.
    robust_metrics = evaluate(
        model,
        seed=args.robust_eval_seed,
        episodes=args.robust_eval_episodes,
        domain_randomization=True,
        terrain_task=args.terrain,
        rough_level=args.rough_level,
    )
    robust_zero_residual_metrics = evaluate(
        None,
        seed=args.robust_eval_seed,
        episodes=args.robust_eval_episodes,
        domain_randomization=True,
        terrain_task=args.terrain,
        rough_level=args.rough_level,
    )
    candidate_profiles = (metrics, robust_metrics)
    baseline_profiles = (zero_residual_metrics, robust_zero_residual_metrics)
    terrain_mode = args.terrain != "flat"
    acceptance = {
        "nominal_and_repeat_root_wrench_zero": all(
            profile["root_wrench_max_abs"] == 0.0 for profile in candidate_profiles
        ),
        "nominal_and_repeat_fall_rate_zero": all(
            profile["fall_rate"] == 0.0 for profile in candidate_profiles
        ),
        "terrain_world_x_progress_ge_7m": all(
            profile["mean_world_x_progress_m"] >= 7.0 for profile in candidate_profiles
        ) if terrain_mode else True,
        "terrain_world_x_progress_p05_ge_6m": all(
            profile["world_x_progress_p05_m"] >= 6.0 for profile in candidate_profiles
        ) if terrain_mode else True,
        "terrain_lateral_offset_p95_le_1m": all(
            profile["course_lateral_offset_p95_m"] <= 1.0 for profile in candidate_profiles
        ) if terrain_mode else True,
        "terrain_heading_error_p95_le_25deg": all(
            profile["course_heading_error_p95_deg"] <= 25.0 for profile in candidate_profiles
        ) if terrain_mode else True,
        "repeat_falls_not_worse_than_zero_residual": (
            robust_metrics["fall_rate"] <= robust_zero_residual_metrics["fall_rate"]
            if terrain_mode
            else robust_metrics["fall_rate"] < robust_zero_residual_metrics["fall_rate"]
        ),
        "velocity_rmse_le_0_75": all(
            profile["time_series_velocity_rmse_mps"] <= (0.75 if terrain_mode else 0.45) for profile in candidate_profiles
        ),
        "yaw_rmse_le_1_00": all(
            profile["time_series_yaw_rate_rmse_radps"] <= (1.00 if terrain_mode else 0.55) for profile in candidate_profiles
        ),
        "slip_le_0_28": all(
            profile["mean_stance_foot_slip_mps"] <= (0.28 if terrain_mode else 0.12) for profile in candidate_profiles
        ),
        "gait_match_ge_0_58": all(
            profile["mean_gait_contact_match"] >= (0.58 if terrain_mode else 0.82) for profile in candidate_profiles
        ),
        "applied_action_saturation_le_0_02": all(
            profile["action_saturation_fraction"] <= 0.02 for profile in candidate_profiles
        ),
        "torque_saturation_le_0_08": all(
            profile["torque_saturation_fraction"] <= (0.08 if terrain_mode else 0.01) for profile in candidate_profiles
        ),
        "base_clearance_p05_ge_0_25": all(
            profile["base_clearance_p05_m"] >= (0.25 if terrain_mode else 0.27) for profile in candidate_profiles
        ),
        "base_clearance_p95_le_0_36": all(
            profile["base_clearance_p95_m"] <= 0.36 for profile in candidate_profiles
        ),
        "base_tilt_p95_le_22deg": all(
            profile["base_tilt_p95_deg"] <= (22.0 if terrain_mode else 10.0) for profile in candidate_profiles
        ),
        "terrain_progress_better_than_zero_residual": all(
            candidate["mean_world_x_progress_m"] >= baseline["mean_world_x_progress_m"] + (1.0 if terrain_mode else 0.0)
            for candidate, baseline in zip(candidate_profiles, baseline_profiles)
        ),
        "heading_not_worse_than_zero_residual": all(
            candidate["course_heading_error_p95_deg"] <= baseline["course_heading_error_p95_deg"]
            for candidate, baseline in zip(candidate_profiles, baseline_profiles)
        ),
        "slip_not_worse_than_zero_residual_5pct": all(
            candidate["mean_stance_foot_slip_mps"] <= 1.05 * baseline["mean_stance_foot_slip_mps"]
            for candidate, baseline in zip(candidate_profiles, baseline_profiles)
        ),
        "gait_not_worse_than_zero_residual": all(
            candidate["mean_gait_contact_match"] >= baseline["mean_gait_contact_match"]
            for candidate, baseline in zip(candidate_profiles, baseline_profiles)
        ),
    }
    acceptance["passed"] = all(acceptance.values())
    manifest: dict[str, Any] = {
        "backend": "MuJoCo",
        "source_repository": "https://github.com/yang-zj1026/legged-loco",
        "source_revision": "87b0d3d18404e784abc0a62227bc41c940f29ecc",
        "source_checkpoint_available": False,
        "reason_for_mujoco_retraining": "Repository provides Isaac Lab training code but no released Go2 checkpoint (.pt/.jit/.onnx).",
        "model": str(canonical_model),
        "candidate_model": str(candidate_model),
        "new_training_timesteps": args.timesteps,
        "total_training_timesteps": int(model.num_timesteps),
        "resumed_from": resumed_from,
        "seed": args.seed,
        "terrain": (
            {"task": "rough", "curriculum_levels": [1, 2, 3], "per_level": [terrain_metadata("rough", level) for level in (1, 2, 3)]}
            if args.terrain == "rough" and args.rough_level is None
            else terrain_metadata(args.terrain, args.rough_level)
        ),
        "controller": {
            "physics_timestep_s": 0.005,
            "control_decimation": 4,
            "control_timestep_s": 0.02,
            "observation_order": "[base_ang_vel(3), base_rpy(3), velocity_command(3), joint_pos_rel(12), joint_vel_rel(12), last_action(12)]",
            "single_observation_dim": 45,
            "history_length": 9,
            "policy_observation_dim": 450,
            "action": (
                "12 normalized PPO residuals are trained around the trot prior; "
                f"terrain deployment applies the same nonzero joint residual gain {TERRAIN_RESIDUAL_POLICY_GAIN:.2f} and raw-action limit {TERRAIN_RESIDUAL_ACTION_LIMIT:.2f} used in training"
                if terrain_mode else
                f"12 normalized PPO residuals, deployment-conditioned by gain {DEPLOYMENT_POLICY_ACTION_GAIN:.2f}, then mapped once by 0.18 rad around the trot prior"
            ),
            "deployment_policy_action_gain": TERRAIN_RESIDUAL_POLICY_GAIN if terrain_mode else DEPLOYMENT_POLICY_ACTION_GAIN,
            "terrain_raw_action_limit": TERRAIN_RESIDUAL_ACTION_LIMIT if terrain_mode else None,
            "pd": {"stiffness": 60.0, "damping": 2.0, "delay_control_steps": 4},
            "trot": {"duty_factor": 0.58, "phase_offsets": [0.0, 0.5, 0.5, 0.0], "foot_target": "sagittal two-link IK with stance-locked world velocity, smooth Hermite swing, and terrain-mode base-IMU joint-reference feedback"},
        },
        "ppo": {
            "actor_hidden_dims": [512, 256, 128],
            "activation": "ELU",
            "learning_rate": float(args.learning_rate) if args.learning_rate is not None else 0.0001,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.10,
            "entropy_coefficient": 0.0,
            "initial_log_std": -3.0,
            "state_dependent_exploration": True,
            "squashed_action_distribution": True,
            "target_kl": 0.012,
            "value_loss_coefficient": 0.5,
            "epochs": 3,
            "max_grad_norm": 0.5,
        },
        "payload": "The exact 0.22 kg rigid Go2 dorsal QR fixture is present during locomotion training.",
        "root_wrench": "disabled: Go2 base xfrc_applied is identically zero; motion and balance come from 12 joint actuators and foot contacts",
        "sim_to_real_randomization": "training only: foot/ground friction, motor strength, PD gains, 3--5 step latency, IMU and joint-encoder noise. Terrain certificate reports fixed-course repeatability, not broad randomization robustness.",
        "reward_additions": "stance-foot slip, 0.32 m base height, diagonal contact schedule, upright body, action magnitude and action-rate penalties",
        "evaluation": metrics,
        "zero_residual_evaluation": zero_residual_metrics,
        "held_out_repeat_evaluation": robust_metrics,
        "held_out_repeat_zero_residual_evaluation": robust_zero_residual_metrics,
        "held_out_repeat_seed": args.robust_eval_seed,
        "held_out_repeat_episodes": args.robust_eval_episodes,
        "acceptance": acceptance,
    }
    if acceptance["passed"]:
        shutil.copy2(candidate_model, canonical_model)
        manifest["promotion"] = "accepted candidate copied to canonical model"
    else:
        manifest["promotion"] = "rejected candidate retained; canonical model not overwritten"
    metrics_path = args.metrics_file or (args.artifacts_dir / "go2_legged_loco_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    env.close()
    print(json.dumps(manifest, indent=2), flush=True)
    if not acceptance["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
