#!/usr/bin/env python3
"""Print a seeded 2 m–7 m QR-annulus start pose for shell launchers."""

from __future__ import annotations

import argparse

from landing_rl.scenario import sample_annulus_start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    start = sample_annulus_start(args.seed)
    print(f"{start.x_m:.6f} {start.y_m:.6f} {start.radius_m:.6f} {start.heading_rad:.6f}")


if __name__ == "__main__":
    main()
