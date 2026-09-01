"""Train PPO, DDPG and SAC directly in the MuJoCo moving-QR environment."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from .mujoco_environment import MujocoQrPrecisionLandingEnv
from .scenario import MOTION_DIFFICULTIES, MOTION_PROFILE_BOUNDS
from .train import ALGORITHMS, make_model


def evaluate_landing(model: Any, *, seed: int, episodes: int, difficulty: str) -> dict[str, float]:
    """Evaluate a policy in fresh MuJoCo worlds and return landing evidence."""
    env = MujocoQrPrecisionLandingEnv(seed=seed, difficulty=difficulty)
    rewards: list[float] = []
    terminal_errors: list[float] = []
    terminal_steps: list[float] = []
    successes = 0
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        total_reward = 0.0
        last_info: dict[str, float] = {}
        while True:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, last_info = env.step(np.asarray(action, dtype=np.float32))
            total_reward += reward
            if terminated or truncated:
                break
        successes += int(last_info.get("success", 0.0) > 0.5)
        rewards.append(total_reward)
        terminal_errors.append(last_info.get("horizontal_error_m", float("inf")))
        terminal_steps.append(last_info.get("episode_steps", float("inf")))
    env.close()
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "success_rate": successes / episodes,
        "mean_terminal_error_m": float(np.mean(terminal_errors)),
        "mean_episode_steps": float(np.mean(terminal_steps)),
        "mean_episode_duration_s": env.dt * float(np.mean(terminal_steps)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithms", default="ppo,ddpg,sac")
    parser.add_argument("--timesteps", type=int, default=60_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/rl_training"))
    parser.add_argument("--model-suffix", default="mujoco_moving_qr")
    parser.add_argument("--metrics-file", default="mujoco_training_metrics.json")
    parser.add_argument("--evaluation-difficulties", default="easy,medium,hard")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = [name.strip().lower() for name in args.algorithms.split(",") if name.strip()]
    if not selected or any(name not in ALGORITHMS for name in selected):
        raise SystemExit(f"--algorithms must be drawn from: {', '.join(ALGORITHMS)}")
    difficulties = [value.strip().lower() for value in args.evaluation_difficulties.split(",") if value.strip()]
    if not difficulties or any(value not in MOTION_DIFFICULTIES or value == "train" for value in difficulties):
        raise SystemExit("--evaluation-difficulties must contain only easy,medium,hard")
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    set_random_seed(args.seed)
    results: dict[str, dict[str, Any]] = {}
    best_name, best_score = "", float("-inf")

    for offset, name in enumerate(selected):
        seed = args.seed + offset
        environment = Monitor(MujocoQrPrecisionLandingEnv(seed=seed, difficulty="train"))
        model = make_model(name, environment, seed)
        model.learn(total_timesteps=args.timesteps, progress_bar=False)
        model_path = args.models_dir / f"{name}_{args.model_suffix}"
        model.save(model_path)
        training = evaluate_landing(model, seed=seed + 10_000, episodes=args.eval_episodes, difficulty="train")
        held_out = {
            difficulty: evaluate_landing(model, seed=seed + 20_000 + index * 1_000, episodes=args.eval_episodes, difficulty=difficulty)
            for index, difficulty in enumerate(difficulties)
        }
        results[name] = {"training": training, "held_out": held_out}
        score_source = held_out.get("medium", next(iter(held_out.values())))
        score = score_source["success_rate"] * 100.0 + score_source["mean_reward"] / 10.0
        if score > best_score:
            best_name, best_score = name, score
        environment.close()
        print(f"{name}: {json.dumps(results[name], sort_keys=True)}", flush=True)

    shutil.copy2(args.models_dir / f"{best_name}_{args.model_suffix}.zip", args.models_dir / f"best_{args.model_suffix}.zip")
    manifest = {
        "backend": "MuJoCo",
        "mujoco_version": mujoco.__version__,
        "seed": args.seed,
        "timesteps_per_algorithm": args.timesteps,
        "algorithms": selected,
        "best_algorithm": best_name,
        "metrics": results,
        "training_difficulty": "train",
        "evaluation_difficulties": difficulties,
        "observation": "[qr_error_x, qr_error_y, altitude, detected, target_velocity_x, target_velocity_y]",
        "action": "[lateral_x, lateral_y] velocity residual over MuJoCo visual servo + moving-pad velocity feed-forward",
        "start_distribution": "drone begins uniformly at a random 2–7 m annulus position around the QR pad",
        "physics": "MuJoCo free-joint quadrotor with gravity, mass/inertia, force-based velocity servo, moving mocap QR deck and ground contact",
        "motion_distributions": {
            name: {"base_speed_mps": values.base_speed_mps, "lateral_amplitude_mps": values.lateral_amplitude_mps}
            for name, values in MOTION_PROFILE_BOUNDS.items()
        },
    }
    (args.artifacts_dir / args.metrics_file).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"best policy: {best_name} -> {args.models_dir / f'best_{args.model_suffix}.zip'}", flush=True)


if __name__ == "__main__":
    main()
