"""Vision-guided precision landing with QR detection and reinforcement learning."""

from .environment import QrPrecisionLandingEnv
from .vision import QrDetection, QrDetector

__all__ = ("QrDetection", "QrDetector", "QrPrecisionLandingEnv")
