#!/usr/bin/env python3
"""Render a physical MuJoCo road made of individually placed gravel rocks.

This is a visual/physical-terrain approval preview.  Every visible stone is a
static MuJoCo ellipsoid collision geom embedded in a compacted soil base; it
is not a texture or a smooth heightfield pretending to be stones.
"""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "artifacts" / "rl_training" / "go2_discrete_gravel_road_preview.png"
SEED = 20260901
ROAD_LENGTH_M = 34.0
ROAD_HALF_WIDTH_M = 2.35
# Long farm track: only 0.8% grade (about 27 cm over 34 m), plus very gentle
# compacted-soil waviness underneath the individually colliding stones.
ROAD_SLOPE_GRADE = 0.008
ROAD_START_X_M = -15.0


def road_height(x_m: float, y_m: float) -> float:
    """A gentle 2.2% grade with compacted-road scale undulation in metres."""
    progress = np.clip((x_m - ROAD_START_X_M) / ROAD_LENGTH_M, 0.0, 1.0)
    return float(
        ROAD_SLOPE_GRADE * max(0.0, x_m - ROAD_START_X_M)
        + 0.010 * np.sin(2.0 * np.pi * progress * 2.1)
        + 0.004 * np.sin(2.0 * np.pi * progress * 6.0 + 1.1)
        + 0.002 * np.cos(2.0 * np.pi * y_m / (2.0 * ROAD_HALF_WIDTH_M))
    )


def rock_xml() -> str:
    """Return dense, varied, individually collidable gravel stones."""
    rng = np.random.default_rng(SEED)
    colors = (
        ".22 .22 .20 1",  # charcoal stone
        ".31 .30 .27 1",  # warm granite
        ".40 .37 .31 1",  # sand stone
        ".20 .24 .24 1",  # dark slate
        ".47 .43 .36 1",  # pale rock
        ".28 .27 .24 1",
    )
    rocks: list[str] = []
    # The reference is a dense river-gravel farm track: three size bands make
    # an unbroken layer of recognisable rounded stones rather than a regular
    # rock grid.  Each remains an individual static collision body.
    bands = ((5700, 0.021, 0.050), (2300, 0.047, 0.082), (390, 0.078, 0.128))
    for band, (count, minimum, maximum) in enumerate(bands):
        for index in range(count):
            px = float(rng.uniform(ROAD_START_X_M, ROAD_START_X_M + ROAD_LENGTH_M))
            py = float(rng.uniform(-ROAD_HALF_WIDTH_M + 0.04, ROAD_HALF_WIDTH_M - 0.04))
            sx = float(rng.uniform(minimum, maximum))
            sy = sx * float(rng.uniform(0.62, 1.05))
            sz = sx * float(rng.uniform(0.34, 0.58))
            # Embed the lower half in compacted soil exactly as a road stone
            # is packed in real gravel.  The exposed part is still collision.
            z = road_height(px, py) - 0.010 + 0.56 * sz
            yaw = float(rng.uniform(-3.1416, 3.1416))
            roll = float(rng.uniform(-0.34, 0.34))
            pitch = float(rng.uniform(-0.30, 0.30))
            color = colors[int(rng.integers(len(colors)))]
            rocks.append(
                f'<geom name="gravel_rock_{band}_{index}" type="ellipsoid" '
                f'pos="{px:.4f} {py:.4f} {z:.4f}" euler="{roll:.4f} {pitch:.4f} {yaw:.4f}" '
                f'size="{sx:.4f} {sy:.4f} {sz:.4f}" rgba="{color}" '
                'friction="1.15 .018 .008" condim="3" contype="1" conaffinity="1"/>'
            )
    return "\n".join(rocks)


def verge_xml() -> str:
    """Return dry grass/reed clusters along both edges, as in the reference."""
    rng = np.random.default_rng(SEED + 1)
    stems: list[str] = []
    colors = (".33 .23 .12 .92", ".45 .31 .16 .90", ".54 .39 .21 .86", ".29 .22 .14 .88")
    for side in (-1.0, 1.0):
        for index in range(430):
            x = float(rng.uniform(ROAD_START_X_M - 0.1, ROAD_START_X_M + ROAD_LENGTH_M + 0.4))
            y = side * float(rng.uniform(ROAD_HALF_WIDTH_M + 0.03, ROAD_HALF_WIDTH_M + 0.90))
            height = float(rng.uniform(0.38, 1.15))
            lean_x = float(rng.uniform(-0.18, 0.18))
            lean_y = side * float(rng.uniform(-0.16, 0.18))
            colour = colors[int(rng.integers(len(colors)))]
            ground_z = road_height(x, side * ROAD_HALF_WIDTH_M)
            stems.append(
                f'<geom name="dry_grass_{int(side)}_{index}" type="capsule" '
                f'fromto="{x:.4f} {y:.4f} {ground_z - .01:.4f} {x + lean_x:.4f} {y + lean_y:.4f} {ground_z + height:.4f}" '
                f'size=".007" rgba="{colour}" contype="0" conaffinity="0"/>'
            )
    return "\n".join(stems)


def road_hfield_xml() -> str:
    """Build the gentle physical soil surface under the discrete stones."""
    x_samples = np.linspace(ROAD_START_X_M, ROAD_START_X_M + ROAD_LENGTH_M, 341)
    y_samples = np.linspace(-ROAD_HALF_WIDTH_M, ROAD_HALF_WIDTH_M, 49)
    heights = np.array([[road_height(float(x), float(y)) for x in x_samples] for y in y_samples])
    lower = float(heights.min())
    upper = float(heights.max())
    elevation = " ".join(f"{value:.7f}" for value in ((heights - lower) / (upper - lower)).ravel())
    return (
        f'<hfield name="compacted_road_hfield" nrow="{len(y_samples)}" ncol="{len(x_samples)}" '
        f'size="{0.5 * ROAD_LENGTH_M:.6f} {ROAD_HALF_WIDTH_M:.6f} {upper - lower:.6f} .18" '
        f'elevation="{elevation}"/>'
    )


def build_xml() -> str:
    return f'''<mujoco model="discrete_gravel_road_preview">
  <compiler angle="radian"/>
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <visual><global offwidth="1600" offheight="900"/><quality shadowsize="4096"/><headlight ambient=".44 .40 .32" diffuse=".55 .50 .42" specular=".10 .10 .10"/><map znear="0.01" zfar="80" haze=".22"/><rgba haze=".18 .25 .34 1" fog=".18 .25 .34 1"/></visual>
  <asset>
    <texture name="skybox" type="skybox" builtin="gradient" rgb1=".56 .65 .72" rgb2=".20 .28 .36" width="512" height="512"/>
    <texture name="soil" type="2d" builtin="flat" mark="random" rgb1=".21 .20 .15" markrgb=".10 .10 .07" random=".18" width="512" height="512"/>
    <material name="soil" texture="soil" texrepeat="8 4" texuniform="true" specular=".04" shininess=".10"/>
    <texture name="road_dirt" type="2d" builtin="flat" mark="random" rgb1=".31 .30 .27" markrgb=".15 .15 .13" random=".22" width="512" height="512"/>
    <material name="road_dirt" texture="road_dirt" texrepeat="12 4" texuniform="true" specular=".02" shininess=".08"/>
    {road_hfield_xml()}
  </asset>
  <worldbody>
    <light name="sun" pos="-4 -6 13" dir=".28 .36 -1" directional="true" diffuse="1 0.93 .78" specular=".12 .12 .12" castshadow="true"/>
    <light name="fill" pos="14 6 8" dir="-.7 -.3 -1" directional="true" diffuse=".36 .46 .58"/>
    <geom name="surrounding_ground" type="plane" pos="0 0 -.20" size="50 50 .1" rgba=".32 .30 .22 1" material="soil" friction="1.0 .02 .01"/>
    <geom name="compacted_road_base" type="hfield" hfield="compacted_road_hfield" pos="{ROAD_START_X_M + 0.5 * ROAD_LENGTH_M:.3f} 0 0" material="road_dirt" rgba=".65 .65 .62 1" friction="1.12 .018 .008" condim="3" contype="1" conaffinity="1"/>
    <geom name="left_soil_edge" type="box" pos="{ROAD_START_X_M + 0.5 * ROAD_LENGTH_M:.3f} {-ROAD_HALF_WIDTH_M - .34:.3f} .08" size="{0.5 * ROAD_LENGTH_M:.3f} .36 .28" rgba=".32 .30 .22 1" material="soil"/>
    <geom name="right_soil_edge" type="box" pos="{ROAD_START_X_M + 0.5 * ROAD_LENGTH_M:.3f} {ROAD_HALF_WIDTH_M + .34:.3f} .08" size="{0.5 * ROAD_LENGTH_M:.3f} .36 .28" rgba=".32 .30 .22 1" material="soil"/>
    {rock_xml()}
    {verge_xml()}
  </worldbody>
</mujoco>'''


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_string(build_xml())
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (3.0, 0.0, road_height(3.0, 0.0) + 0.15)
    camera.distance = 17.0
    camera.azimuth = 180.0
    camera.elevation = -10.0
    renderer = mujoco.Renderer(model, width=1600, height=900)
    renderer.update_scene(data, camera=camera)
    imageio.imwrite(OUTPUT, renderer.render())
    renderer.close()
    print(OUTPUT)


if __name__ == "__main__":
    main()
