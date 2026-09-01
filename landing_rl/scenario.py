"""Shared geometry and motion profiles for the QR landing scenarios.

The Gazebo world is centered on the QR landing pad at ``(0, 0)``.  Vehicles
begin on the ground at a uniformly sampled point in the open annulus between
the 2 m safety ring and the 7 m outer-search ring.  A seed makes each filmed
run reproducible while retaining the same random sampling rule.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass


QR_SIZE_M = 0.40
INNER_RING_RADIUS_M = 2.0
OUTER_RING_RADIUS_M = 7.0
# The moving-pad experiment uses a deliberately modest speed: it is fast
# enough to require a sustained visual track during descent, while remaining
# inside a 0.4 m pad's physical landing corridor after the final land command.
MOVING_QR_SPEED_MPS = 0.08
MOVING_QR_HEADING_DEG = 0.0
# The accelerated experiment keeps exactly the same smooth geometry as the
# previous wavy target, but triples every horizontal velocity component.
WAVY_QR_SPEED_MULTIPLIER = 3.0
MOTION_DIFFICULTIES = ("train", "easy", "medium", "hard")


@dataclass(frozen=True)
class WavyMotionProfile:
    """A seeded, smooth target path plus a small boat-like deck motion.

    Each profile is intentionally bounded: it follows a broad sinusoidal
    curve instead of making discontinuous turns, while speed varies in a
    narrow range suitable for a 0.4 m visual landing target.  The same
    equations are evaluated by the training environment, online controller,
    and Gazebo moving-platform plugin.
    """

    seed: int
    heading_deg: float
    base_speed_mps: float
    speed_amplitude_mps: float
    speed_frequency_radps: float
    speed_phase_rad: float
    lateral_amplitude_mps: float
    turn_frequency_radps: float
    turn_phase_rad: float
    wave_roll_deg: float
    wave_pitch_deg: float
    wave_yaw_deg: float
    wave_heave_m: float
    wave_roll_frequency_radps: float
    wave_pitch_frequency_radps: float
    wave_yaw_frequency_radps: float
    wave_heave_frequency_radps: float
    wave_phase_rad: float

    def velocity_at(self, motion_time_s: float) -> tuple[float, float]:
        """Return instantaneous target velocity in Gazebo ENU metres/s."""
        t = max(0.0, motion_time_s)
        forward = self.base_speed_mps + self.speed_amplitude_mps * math.sin(
            self.speed_frequency_radps * t + self.speed_phase_rad
        )
        right = self.lateral_amplitude_mps * math.sin(
            self.turn_frequency_radps * t + self.turn_phase_rad
        )
        heading = math.radians(self.heading_deg)
        return (
            math.cos(heading) * forward - math.sin(heading) * right,
            math.sin(heading) * forward + math.cos(heading) * right,
        )

    def position_at(self, motion_time_s: float) -> tuple[float, float]:
        """Integrate the smooth velocity profile from the initial QR origin."""
        t = max(0.0, motion_time_s)
        forward = self.base_speed_mps * t
        forward += self.speed_amplitude_mps / self.speed_frequency_radps * (
            math.cos(self.speed_phase_rad)
            - math.cos(self.speed_frequency_radps * t + self.speed_phase_rad)
        )
        right = self.lateral_amplitude_mps / self.turn_frequency_radps * (
            math.cos(self.turn_phase_rad)
            - math.cos(self.turn_frequency_radps * t + self.turn_phase_rad)
        )
        heading = math.radians(self.heading_deg)
        return (
            math.cos(heading) * forward - math.sin(heading) * right,
            math.sin(heading) * forward + math.cos(heading) * right,
        )

    def wave_at(self, motion_time_s: float) -> tuple[float, float, float, float]:
        """Return roll/pitch/yaw in degrees and deck heave in metres."""
        t = max(0.0, motion_time_s)
        phase = self.wave_phase_rad
        return (
            self.wave_roll_deg * math.sin(self.wave_roll_frequency_radps * t + phase),
            self.wave_pitch_deg * math.sin(self.wave_pitch_frequency_radps * t + phase + 0.83),
            self.wave_yaw_deg * math.sin(self.wave_yaw_frequency_radps * t + phase + 1.71),
            self.wave_heave_m * math.sin(self.wave_heave_frequency_radps * t + phase + 2.41),
        )

    def shell_values(self) -> str:
        """Stable positional representation consumed by ``run_all.sh``."""
        values = (
            self.seed,
            self.heading_deg,
            self.base_speed_mps,
            self.speed_amplitude_mps,
            self.speed_frequency_radps,
            self.speed_phase_rad,
            self.lateral_amplitude_mps,
            self.turn_frequency_radps,
            self.turn_phase_rad,
            self.wave_roll_deg,
            self.wave_pitch_deg,
            self.wave_yaw_deg,
            self.wave_heave_m,
            self.wave_roll_frequency_radps,
            self.wave_pitch_frequency_radps,
            self.wave_yaw_frequency_radps,
            self.wave_heave_frequency_radps,
            self.wave_phase_rad,
        )
        return " ".join(f"{float(value):.8f}" if index else str(int(value)) for index, value in enumerate(values))


@dataclass(frozen=True)
class MotionProfileBounds:
    """Independent trajectory distributions for learning and evaluation."""

    base_speed_mps: tuple[float, float]
    speed_amplitude_mps: tuple[float, float]
    speed_frequency_radps: tuple[float, float]
    lateral_amplitude_mps: tuple[float, float]
    turn_frequency_radps: tuple[float, float]
    roll_deg: tuple[float, float]
    pitch_deg: tuple[float, float]
    yaw_deg: tuple[float, float]
    heave_m: tuple[float, float]
    wave_frequency_radps: tuple[float, float]


# Training deliberately uses short, calm terminal-landing motion.  Evaluation
# profiles have disjoint speed ranges and progressively longer/faster curves,
# so a model cannot be credited for replaying its training trajectory.
MOTION_PROFILE_BOUNDS: dict[str, MotionProfileBounds] = {
    "train": MotionProfileBounds(
        base_speed_mps=(0.035, 0.060), speed_amplitude_mps=(0.003, 0.009),
        speed_frequency_radps=(0.08, 0.13), lateral_amplitude_mps=(0.006, 0.016),
        turn_frequency_radps=(0.10, 0.16), roll_deg=(0.25, 0.65), pitch_deg=(0.20, 0.55),
        yaw_deg=(0.30, 0.75), heave_m=(0.002, 0.007), wave_frequency_radps=(0.35, 0.62),
    ),
    "easy": MotionProfileBounds(
        base_speed_mps=(0.115, 0.155), speed_amplitude_mps=(0.012, 0.024),
        speed_frequency_radps=(0.07, 0.12), lateral_amplitude_mps=(0.022, 0.045),
        turn_frequency_radps=(0.09, 0.15), roll_deg=(0.55, 1.05), pitch_deg=(0.45, 0.95),
        yaw_deg=(0.60, 1.25), heave_m=(0.006, 0.012), wave_frequency_radps=(0.45, 0.80),
    ),
    "medium": MotionProfileBounds(
        base_speed_mps=(0.234, 0.306), speed_amplitude_mps=(0.024, 0.051),
        speed_frequency_radps=(0.11, 0.17), lateral_amplitude_mps=(0.072, 0.123),
        turn_frequency_radps=(0.16, 0.24), roll_deg=(1.10, 2.00), pitch_deg=(1.00, 1.80),
        yaw_deg=(1.40, 2.60), heave_m=(0.010, 0.020), wave_frequency_radps=(0.48, 1.10),
    ),
    "hard": MotionProfileBounds(
        base_speed_mps=(0.460, 0.560), speed_amplitude_mps=(0.055, 0.095),
        speed_frequency_radps=(0.15, 0.23), lateral_amplitude_mps=(0.135, 0.215),
        turn_frequency_radps=(0.22, 0.34), roll_deg=(1.80, 3.20), pitch_deg=(1.60, 2.90),
        yaw_deg=(2.20, 4.00), heave_m=(0.018, 0.035), wave_frequency_radps=(0.68, 1.35),
    ),
}


def random_motion_profile(seed: int, difficulty: str = "medium") -> WavyMotionProfile:
    """Create one reproducible profile from a named, disjoint distribution."""
    if difficulty not in MOTION_PROFILE_BOUNDS:
        raise ValueError(f"difficulty must be one of: {', '.join(MOTION_DIFFICULTIES)}")
    generator = random.Random(seed)
    bounds = MOTION_PROFILE_BOUNDS[difficulty]
    return WavyMotionProfile(
        seed=seed,
        heading_deg=generator.uniform(-150.0, 150.0),
        base_speed_mps=generator.uniform(*bounds.base_speed_mps),
        speed_amplitude_mps=generator.uniform(*bounds.speed_amplitude_mps),
        speed_frequency_radps=generator.uniform(*bounds.speed_frequency_radps),
        speed_phase_rad=generator.uniform(-math.pi, math.pi),
        lateral_amplitude_mps=generator.uniform(*bounds.lateral_amplitude_mps),
        turn_frequency_radps=generator.uniform(*bounds.turn_frequency_radps),
        turn_phase_rad=generator.uniform(-math.pi, math.pi),
        wave_roll_deg=generator.uniform(*bounds.roll_deg),
        wave_pitch_deg=generator.uniform(*bounds.pitch_deg),
        wave_yaw_deg=generator.uniform(*bounds.yaw_deg),
        wave_heave_m=generator.uniform(*bounds.heave_m),
        wave_roll_frequency_radps=generator.uniform(*bounds.wave_frequency_radps),
        wave_pitch_frequency_radps=generator.uniform(*bounds.wave_frequency_radps),
        wave_yaw_frequency_radps=generator.uniform(*bounds.wave_frequency_radps),
        wave_heave_frequency_radps=generator.uniform(*bounds.wave_frequency_radps),
        wave_phase_rad=generator.uniform(-math.pi, math.pi),
    )


def random_training_motion_profile(seed: int) -> WavyMotionProfile:
    """Short, calm terminal-landing distribution used only while training."""
    return random_motion_profile(seed, "train")


def random_evaluation_motion_profile(seed: int, difficulty: str = "medium") -> WavyMotionProfile:
    """Longer, faster, out-of-distribution profile used for evaluation/SITL."""
    if difficulty == "train":
        raise ValueError("evaluation difficulty must be easy, medium, or hard")
    return random_motion_profile(seed, difficulty)


def random_wavy_motion_profile(seed: int) -> WavyMotionProfile:
    """Backward-compatible medium evaluation profile for existing callers."""
    return random_evaluation_motion_profile(seed, "medium")


@dataclass(frozen=True)
class AnnulusStart:
    """A ground-start pose relative to the QR center, in Gazebo ENU metres."""

    x_m: float
    y_m: float
    radius_m: float
    heading_rad: float
    seed: int


def sample_annulus_start(seed: int) -> AnnulusStart:
    """Sample one strictly inside the 2 m–7 m landing-search annulus."""
    generator = random.Random(seed)
    # Avoid an exact paint-line spawn while retaining a uniform radius rule
    # over the requested [2 m, 7 m] interval.
    radius = generator.uniform(INNER_RING_RADIUS_M + 0.01, OUTER_RING_RADIUS_M - 0.01)
    heading = generator.uniform(-math.pi, math.pi)
    return AnnulusStart(
        x_m=radius * math.cos(heading),
        y_m=radius * math.sin(heading),
        radius_m=radius,
        heading_rad=heading,
        seed=seed,
    )


def moving_qr_velocity(heading_deg: float = MOVING_QR_HEADING_DEG, speed_mps: float = MOVING_QR_SPEED_MPS) -> tuple[float, float]:
    """Return the fixed Gazebo-ENU velocity vector for the moving QR pad."""
    heading = math.radians(heading_deg)
    return speed_mps * math.cos(heading), speed_mps * math.sin(heading)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample a reproducible QR-annulus vehicle spawn.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--wavy-motion-profile",
        action="store_true",
        help="Print the seeded smooth trajectory/deck-motion parameters for Gazebo.",
    )
    parser.add_argument("--motion-profile", choices=MOTION_DIFFICULTIES, default="medium")
    args = parser.parse_args()
    if args.wavy_motion_profile:
        print(random_motion_profile(args.seed, args.motion_profile).shell_values())
        return
    start = sample_annulus_start(args.seed)
    print(f"{start.x_m:.6f} {start.y_m:.6f} {start.radius_m:.6f} {start.heading_rad:.6f}")


if __name__ == "__main__":
    main()
