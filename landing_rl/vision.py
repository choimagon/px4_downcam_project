"""OpenCV QR detection shared by training validation and ROS 2 inference."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class QrDetection:
    """A decoded QR target expressed in image coordinates."""

    payload: str
    center_px: tuple[float, float]
    normalized_error: tuple[float, float]
    corners_px: np.ndarray


class QrDetector:
    """Detect only a decoded QR landing pad, never an arbitrary square marker."""

    def __init__(self, expected_payload: str = "QR") -> None:
        self.expected_payload = expected_payload
        self._detector = cv2.QRCodeDetector()

    def detect(self, frame_bgr: np.ndarray) -> QrDetection | None:
        """Return the target center normalized to [-1, 1] or ``None``."""
        if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            return None

        # Gazebo can transiently deliver a partially rendered QR at the image
        # boundary during the acquisition sweep. OpenCV raises for a few such
        # degenerate contours; treating that frame as a miss keeps the flight
        # in its deliberate search state instead of aborting the mission.
        try:
            payload, corners, _ = self._detector.detectAndDecode(frame_bgr)
        except cv2.error:
            return None
        if not payload or corners is None:
            return None
        if self.expected_payload and payload != self.expected_payload:
            return None

        corners_array = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
        center_x, center_y = corners_array.mean(axis=0)
        height, width = frame_bgr.shape[:2]
        error_x = (float(center_x) - width * 0.5) / (width * 0.5)
        error_y = (float(center_y) - height * 0.5) / (height * 0.5)
        return QrDetection(
            payload=payload,
            center_px=(float(center_x), float(center_y)),
            normalized_error=(float(np.clip(error_x, -1.0, 1.0)), float(np.clip(error_y, -1.0, 1.0))),
            corners_px=corners_array,
        )
