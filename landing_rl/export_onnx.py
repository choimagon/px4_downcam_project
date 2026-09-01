"""Export deterministic Stable-Baselines3 landing policies to ONNX."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from stable_baselines3 import DDPG, PPO, SAC


class PpoActor(torch.nn.Module):
    """Deterministic (mean) PPO action used by ``model.predict``."""

    def __init__(self, policy: torch.nn.Module) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        features = self.policy.extract_features(observation)
        latent = self.policy.mlp_extractor.forward_actor(features)
        return self.policy.action_net(latent)


class SacActor(torch.nn.Module):
    def __init__(self, actor: torch.nn.Module) -> None:
        super().__init__()
        self.actor = actor

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.actor(observation, deterministic=True)


def load_model(path: Path):
    errors: list[str] = []
    for algorithm in (PPO, DDPG, SAC):
        try:
            return algorithm.load(path), algorithm.__name__.lower()
        except Exception as error:
            errors.append(f"{algorithm.__name__}: {error}")
    raise RuntimeError(f"Cannot load {path}: {'; '.join(errors)}")


def actor_for(model, name: str) -> torch.nn.Module:
    if name == "ppo":
        return PpoActor(model.policy)
    if name == "ddpg":
        return model.policy.actor
    if name == "sac":
        return SacActor(model.policy.actor)
    raise ValueError(name)


def export(model_path: Path, destination: Path) -> dict[str, object]:
    model, name = load_model(model_path)
    actor = actor_for(model, name).cpu().eval()
    observation_shape = tuple(int(value) for value in model.observation_space.shape)
    if len(observation_shape) != 1:
        raise RuntimeError(f"Only flat observations can be exported: {model_path} -> {observation_shape}")
    observation_size = observation_shape[0]
    example = torch.zeros((1, observation_size), dtype=torch.float32)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            actor,
            example,
            str(destination),
            input_names=["observation"],
            output_names=["action"],
            dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
    onnx.checker.check_model(onnx.load(destination))
    # A deterministic in-range validation input works for both the original
    # 6-state X500 task and the 7-state Go2 camera/PX4-estimator task.
    sample = np.zeros((1, observation_size), dtype=np.float32)
    flat_low = np.asarray(model.observation_space.low, dtype=np.float32).reshape(-1)
    flat_high = np.asarray(model.observation_space.high, dtype=np.float32).reshape(-1)
    finite = np.isfinite(flat_low) & np.isfinite(flat_high)
    sample[0, finite] = 0.18 * flat_low[finite] + 0.82 * flat_high[finite]
    sb3_action, _ = model.predict(sample, deterministic=True)
    session = ort.InferenceSession(str(destination), providers=["CPUExecutionProvider"])
    onnx_action = session.run(["action"], {"observation": sample})[0]
    # SB3 clips the PPO distribution output to the action-space bounds.  The
    # exported model is intentionally raw so runtime callers apply the same
    # clip before passing it to the environment.
    sb3_action = np.clip(np.asarray(sb3_action, dtype=np.float32), -1.0, 1.0)
    max_abs_error = float(np.max(np.abs(sb3_action - onnx_action)))
    if max_abs_error > 2e-5:
        raise RuntimeError(f"ONNX action mismatch for {model_path}: {max_abs_error:.7f}")
    return {
        "algorithm": name,
        "source": str(model_path),
        "onnx": str(destination),
        "input": {"name": "observation", "shape": ["batch", observation_size], "dtype": "float32"},
        "output": {"name": "action", "shape": ["batch", int(model.action_space.shape[0])], "dtype": "float32"},
        "opset": 17,
        "validation_max_abs_action_error": max_abs_error,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", type=Path, help="SB3 .zip policy files")
    parser.add_argument("--output-dir", type=Path, default=Path("models/onnx"))
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/rl_training/mujoco_onnx_models.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifests: list[dict[str, object]] = []
    for model_path in args.models:
        stem = model_path.name.removesuffix(".zip")
        output = args.output_dir / f"{stem}.onnx"
        manifest = export(model_path, output)
        manifests.append(manifest)
        print(json.dumps(manifest, sort_keys=True), flush=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps({"models": manifests}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
