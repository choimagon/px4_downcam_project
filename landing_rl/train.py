"""Train and compare PPO, DDPG, and SAC QR precision-landing policies."""

from __future__ import annotations

import argparse
import json
import shutil
import warnings
from pathlib import Path
from typing import Any

import numpy as np

warnings.filterwarnings("ignore", message="Unable to import Axes3D")

from stable_baselines3 import DDPG, PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.utils import set_random_seed

from .environment import QrPrecisionLandingEnv
from .scenario import MOTION_DIFFICULTIES, MOTION_PROFILE_BOUNDS

ALGORITHMS = {"ppo": PPO, "ddpg": DDPG, "sac": SAC}


def make_model(name: str, env: Monitor, seed: int):
    if name == "ppo":
        return PPO("MlpPolicy", env, n_steps=512, batch_size=128, learning_rate=2.5e-4, gamma=0.997, gae_lambda=0.96, seed=seed, verbose=0)
    if name == "ddpg":
        action_dimensions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(action_dimensions),
            sigma=0.18 * np.ones(action_dimensions),
        )
        return DDPG(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=180_000,
            learning_starts=2_000,
            batch_size=256,
            train_freq=(1, "step"),
            gradient_steps=1,
            gamma=0.997,
            tau=0.01,
            action_noise=action_noise,
            policy_kwargs={"net_arch": [256, 256]},
            seed=seed,
            verbose=0,
        )
    if name == "sac":
        return SAC(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=180_000,
            learning_starts=2_000,
            batch_size=256,
            train_freq=(1, "step"),
            gradient_steps=1,
            gamma=0.997,
            tau=0.01,
            ent_coef="auto_0.02",
            policy_kwargs={"net_arch": [256, 256]},
            seed=seed,
            verbose=0,
        )
    raise ValueError(f"Unsupported algorithm: {name}")


def evaluate_landing(model: Any, seed: int, episodes: int, difficulty: str) -> dict[str, float]:
    """Evaluate on a named held-out motion distribution."""
    env = Monitor(QrPrecisionLandingEnv(seed=seed + 10_000, difficulty=difficulty))
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=episodes, deterministic=True)
    successes = 0
    terminal_errors: list[float] = []
    terminal_steps: list[float] = []
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + 20_000 + episode)
        done = False
        last_info: dict[str, float] = {}
        while not done:
            action, _ = model.predict(observation, deterministic=True)
            # The learned output is a residual to a visual safety controller,
            # not authority to override it. Match the bounded residual used
            # online so offline evaluation measures the deployed policy.
            safe_residual = np.clip(np.asarray(action, dtype=np.float32), -0.25, 0.25)
            observation, _, terminated, truncated, last_info = env.step(safe_residual)
            done = terminated or truncated
        successes += int(last_info.get("success", 0.0) > 0.5)
        terminal_errors.append(last_info.get("horizontal_error_m", float("inf")))
        terminal_steps.append(last_info.get("episode_steps", float("inf")))
    env.close()
    return {
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
        "success_rate": successes / episodes,
        "mean_terminal_error_m": sum(terminal_errors) / len(terminal_errors),
        "mean_episode_steps": sum(terminal_steps) / len(terminal_steps),
        "mean_episode_duration_s": env.unwrapped.dt * sum(terminal_steps) / len(terminal_steps),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithms", default="ppo,ddpg,sac", help="Comma-separated subset of ppo,ddpg,sac")
    parser.add_argument("--timesteps", type=int, default=50_000, help="Training steps per algorithm")
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/rl_training"))
    parser.add_argument("--model-suffix", default="qr_landing", help="Suffix used for per-algorithm model archives")
    parser.add_argument("--metrics-file", default="training_metrics.json", help="Metrics filename inside --artifacts-dir")
    parser.add_argument("--training-difficulty", choices=("train",), default="train")
    parser.add_argument(
        "--evaluation-difficulties",
        default="easy,medium,hard",
        help="Comma-separated held-out difficulties; must not include train.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = [name.strip().lower() for name in args.algorithms.split(",") if name.strip()]
    invalid = sorted(set(selected) - set(ALGORITHMS))
    if invalid or not selected:
        raise SystemExit(f"Algorithms must be drawn from: {', '.join(ALGORITHMS)}")
    evaluation_difficulties = [item.strip().lower() for item in args.evaluation_difficulties.split(",") if item.strip()]
    if not evaluation_difficulties or any(item not in MOTION_DIFFICULTIES or item == "train" for item in evaluation_difficulties):
        raise SystemExit("--evaluation-difficulties must be drawn from: easy, medium, hard")

    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    set_random_seed(args.seed)
    results: dict[str, dict[str, Any]] = {}
    best_name = ""
    best_score = float("-inf")

    for offset, name in enumerate(selected):
        seed = args.seed + offset
        env = Monitor(QrPrecisionLandingEnv(seed=seed, difficulty=args.training_difficulty))
        model = make_model(name, env, seed)
        model.learn(total_timesteps=args.timesteps, progress_bar=False)
        model_path = args.models_dir / f"{name}_{args.model_suffix}"
        model.save(model_path)
        training_metrics = evaluate_landing(model, seed, args.eval_episodes, args.training_difficulty)
        held_out_metrics = {
            difficulty: evaluate_landing(model, seed, args.eval_episodes, difficulty)
            for difficulty in evaluation_difficulties
        }
        results[name] = {"training": training_metrics, "held_out": held_out_metrics}
        score_basis = held_out_metrics.get("medium", next(iter(held_out_metrics.values())))
        score = score_basis["success_rate"] * 100.0 + score_basis["mean_reward"] / 10.0
        if score > best_score:
            best_name, best_score = name, score
        env.close()
        print(f"{name}: {json.dumps(results[name], sort_keys=True)}")

    shutil.copy2(args.models_dir / f"{best_name}_{args.model_suffix}.zip", args.models_dir / f"best_{args.model_suffix}.zip")
    manifest = {
        "seed": args.seed,
        "timesteps_per_algorithm": args.timesteps,
        "algorithms": selected,
        "best_algorithm": best_name,
        "metrics": results,
        "training_difficulty": args.training_difficulty,
        "evaluation_difficulties": evaluation_difficulties,
        "motion_distributions": {
            name: {
                "base_speed_mps": bounds.base_speed_mps,
                "speed_amplitude_mps": bounds.speed_amplitude_mps,
                "lateral_amplitude_mps": bounds.lateral_amplitude_mps,
                "turn_frequency_radps": bounds.turn_frequency_radps,
            }
            for name, bounds in MOTION_PROFILE_BOUNDS.items()
        },
        "observation": "[qr_error_x, qr_error_y, altitude, detected, target_velocity_x, target_velocity_y]",
        "target_domain_randomization": "training uses a short calm terminal-landing distribution; held-out easy/medium/hard inference uses disjoint, longer/faster curved trajectories and deck waves",
        "action": "[lateral_x, lateral_y] residual over visual tracking + instantaneous target-velocity feed-forward; guarded descent starts after stable QR alignment",
    }
    (args.artifacts_dir / args.metrics_file).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"best policy: {best_name} -> {args.models_dir / f'best_{args.model_suffix}.zip'}")


if __name__ == "__main__":
    main()
