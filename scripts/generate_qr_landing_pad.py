#!/usr/bin/env python3
"""Generate the physical QR texture used by the Gazebo landing-pad model."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser()
    # Keep a version-1 QR so the half-size (0.4 m) physical pad remains
    # decodable from the real Gazebo down-camera stream.
    parser.add_argument("--payload", default="QR")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("PX4-Autopilot/Tools/simulation/gz/models/qr_landing_pad/qr_landing_pad.png"),
    )
    args = parser.parse_args()
    qr = cv2.QRCodeEncoder_create().encode(args.payload)
    qr = cv2.copyMakeBorder(qr, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255)
    texture = cv2.resize(qr, (1024, 1024), interpolation=cv2.INTER_NEAREST)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), texture):
        raise RuntimeError(f"Failed to write {args.output}")
    detector = cv2.QRCodeDetector()
    decoded, _, _ = detector.detectAndDecode(texture)
    if decoded != args.payload:
        raise RuntimeError("Generated QR texture did not decode to the expected payload")
    print(f"Generated and verified {args.output}: {decoded}")


if __name__ == "__main__":
    main()
