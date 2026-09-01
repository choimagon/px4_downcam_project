#!/usr/bin/env python3
"""Generate visible 2 m and 7 m annulus rings for the QR Gazebo world.

Each painted ring is assembled from short, collision-free visual segments.  It
avoids relying on renderer-specific torus support and remains clear in the
third-person Gazebo recording.
"""

from __future__ import annotations

import math
from pathlib import Path

from landing_rl.scenario import INNER_RING_RADIUS_M, OUTER_RING_RADIUS_M


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "PX4-Autopilot/Tools/simulation/gz/models/landing_zone_rings"
SEGMENTS = 32


def visual_segment(name: str, radius: float, index: int, rgba: str) -> str:
    angle = 2.0 * math.pi * index / SEGMENTS
    x = radius * math.cos(angle)
    y = radius * math.sin(angle)
    arc_length = 2.0 * radius * math.sin(math.pi / SEGMENTS) * 1.04
    return f"""      <visual name=\"{name}_{index:02d}\">
        <pose>{x:.6f} {y:.6f} 0.012 0 0 {angle:.6f}</pose>
        <geometry><box><size>0.055 {arc_length:.6f} 0.020</size></box></geometry>
        <material><ambient>{rgba}</ambient><diffuse>{rgba}</diffuse><specular>0 0 0 1</specular></material>
      </visual>"""


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_config = """<?xml version=\"1.0\"?>
<model>
  <name>landing_zone_rings</name>
  <version>1.0</version>
  <sdf version=\"1.9\">model.sdf</sdf>
  <author><name>PX4 down-camera project</name></author>
  <description>Visible 2 m inner and 7 m outer QR landing annulus rings.</description>
</model>
"""
    visuals = []
    for index in range(SEGMENTS):
        visuals.append(visual_segment("inner_2m", INNER_RING_RADIUS_M, index, "0.05 0.45 1.0 1"))
        visuals.append(visual_segment("outer_7m", OUTER_RING_RADIUS_M, index, "1.0 0.68 0.05 1"))
    model_sdf = """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<sdf version=\"1.9\">
  <model name=\"landing_zone_rings\">
    <static>true</static>
    <link name=\"rings\">
%s
    </link>
  </model>
</sdf>
""" % "\n".join(visuals)
    (MODEL_DIR / "model.config").write_text(model_config, encoding="utf-8")
    (MODEL_DIR / "model.sdf").write_text(model_sdf, encoding="utf-8")
    print(f"Generated {SEGMENTS}-segment 2 m and 7 m rings in {MODEL_DIR}")


if __name__ == "__main__":
    main()
