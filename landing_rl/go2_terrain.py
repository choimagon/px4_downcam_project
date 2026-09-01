"""Physical terrain definitions shared by Go2 locomotion and landing tasks.

The terrain is deliberately made from MuJoCo collision geoms rather than a
camera-only background or a scripted root trajectory.  Go2 therefore has to
make progress through its 12 actuators and its feet' real contact with the
same slope/rough surface that is visible in the landing recordings.
"""

from __future__ import annotations

import math
from typing import Final

import mujoco
import numpy as np


TERRAIN_TASKS: Final = ("flat", "slope_up", "slope_down", "rough")
TERRAIN_SCENARIOS: Final = (
    ("slope_up", None),
    ("slope_down", None),
    ("rough", 1),
    ("rough", 2),
    ("rough", 3),
)

# The current requirement is a 10 percent physical grade, not a 10 degree
# grade.  tan(theta)=0.10, so theta is about 5.71 degrees.
SLOPE_GRADE: Final = 0.10
SLOPE_GRADE_PERCENT: Final = 100.0 * SLOPE_GRADE
SLOPE_ANGLE_RAD: Final = math.atan(SLOPE_GRADE)
SLOPE_ANGLE_DEG: Final = math.degrees(SLOPE_ANGLE_RAD)
SLOPE_START_X_M: Final = -1.0
# The visible and colliding inclined surface is 16 m long (-1 m ≤ x ≤ 15 m).
# A traversal ends at x=12 m before the physical edge; the robot must not be
# driven off the end of a collision geom merely to fill a 15-second clip.
SLOPE_LENGTH_M: Final = 16.0
SLOPE_HALF_WIDTH_M: Final = 1.70
SLOPE_HALF_THICKNESS_M: Final = 0.30

# The physical course is a 16.0 x 2.4 m MuJoCo heightfield.  It is sampled
# densely enough for compliant continuous contact, rather than being a visual
# mat placed over a separate flat collision plane.
ROUGH_START_X_M: Final = -0.60
ROUGH_TILE_LENGTH_M: Final = 0.42
ROUGH_TILE_COUNT_X: Final = 38
ROUGH_START_Y_M: Final = -1.20
ROUGH_TILE_WIDTH_M: Final = 0.40
ROUGH_TILE_COUNT_Y: Final = 6
ROUGH_BASE_HEIGHT_M: Final = 0.095
ROUGH_TILE_HALF_THICKNESS_M: Final = 0.16
# The former level-3 (24 mm) is now entry level; level 3 is substantially
# harsher while all tiles remain above the surrounding ground plane.
ROUGH_LEVEL_AMPLITUDE_M: Final = {1: 0.024, 2: 0.048, 3: 0.080}
ROUGH_HFIELD_NAME: Final = "terrain_rough_hfield"
ROUGH_HFIELD_GEOM_NAME: Final = "terrain_rough"
# Collision samples are much denser than the visual course cells.  This makes
# the 80 mm terrain a continuous uneven floor instead of a row of vertical
# box walls that a Go2 foot cannot physically traverse.
ROUGH_HFIELD_SAMPLES_X: Final = 161
ROUGH_HFIELD_SAMPLES_Y: Final = 25
TERRAIN_SPEED_MULTIPLIER: Final = 3.0
TERRAIN_ROUTE_TARGET_X_M: Final = 12.0


def validate_terrain_task(task: str) -> str:
    if task not in TERRAIN_TASKS:
        raise ValueError(f"terrain task must be one of: {', '.join(TERRAIN_TASKS)}")
    return task


def validate_rough_level(level: int | None) -> int:
    if level not in (1, 2, 3):
        raise ValueError("rough terrain level must be 1, 2, or 3")
    return int(level)


def terrain_display_name(task: str, level: int | None = None) -> str:
    validate_terrain_task(task)
    if task == "slope_up":
        return f"경사 상승 ({SLOPE_GRADE_PERCENT:g}% · {SLOPE_ANGLE_DEG:.2f}°)"
    if task == "slope_down":
        return f"경사 하강 ({SLOPE_GRADE_PERCENT:g}% · {SLOPE_ANGLE_DEG:.2f}°)"
    if task == "rough":
        return f"울퉁불퉁 지형 {validate_rough_level(level)}단계"
    return "평지"


def terrain_initial_pitch_rad(task: str) -> float:
    """Return the statically valid Go2 base pitch at the terrain start.

    The free base is not kinematically constrained after reset.  This only
    avoids dropping a horizontally initialized robot across a 10 percent
    physical surface at t=0; subsequent pitch and progress come from foot
    contact and the 12 joint actuators.
    """
    validate_terrain_task(task)
    if task == "slope_up":
        return -SLOPE_ANGLE_RAD
    if task == "slope_down":
        return SLOPE_ANGLE_RAD
    return 0.0


def _rough_pattern(index_x: float, index_y: float) -> float:
    """Return bounded, foot-scale continuous roughness in ``[-1, 1]``.

    ``index_x`` and ``index_y`` are nominal 420 x 400 mm course-cell
    coordinates.  The field is deliberately low-gradient so the verified Go2
    gait can traverse the physical 24/48/80 mm collision relief across all
    PPO/DDPG/SAC landing replays.  The renderer exposes it with a close
    foot-level camera and natural granular material, never a camera-only
    checker overlay or a terrain bypass.
    """
    x_m = float(index_x) * ROUGH_TILE_LENGTH_M
    y_m = float(index_y) * ROUGH_TILE_WIDTH_M
    raw = (
        0.58 * math.sin(0.17 * index_x + 0.13 * index_y)
        + 0.29 * math.cos(0.12 * index_x - 0.19 * index_y)
        + 0.13 * math.sin(0.26 * index_x + 0.10 * index_y)
    )
    # The component weights sum to 0.94.  Retain a strict bound against
    # floating-point drift without flattening ordinary peaks through clipping.
    return float(np.clip(raw, -1.0, 1.0))


def rough_tile_height(index_x: int, index_y: int, level: int) -> float:
    """Return the collision top height for one rough-terrain tile."""
    amplitude = ROUGH_LEVEL_AMPLITUDE_M[validate_rough_level(level)]
    return ROUGH_BASE_HEIGHT_M + amplitude * _rough_pattern(index_x, index_y)


def _rough_height_at(x_m: float, y_m: float, level: int) -> float:
    """Continuous physical height of the rough hfield at a world location."""
    index_x = (x_m - ROUGH_START_X_M) / ROUGH_TILE_LENGTH_M
    index_y = (y_m - ROUGH_START_Y_M) / ROUGH_TILE_WIDTH_M
    return ROUGH_BASE_HEIGHT_M + ROUGH_LEVEL_AMPLITUDE_M[validate_rough_level(level)] * _rough_pattern(index_x, index_y)


def _rough_indices(x_m: float, y_m: float) -> tuple[int, int]:
    index_x = int(math.floor((x_m - ROUGH_START_X_M) / ROUGH_TILE_LENGTH_M))
    index_y = int(math.floor((y_m - ROUGH_START_Y_M) / ROUGH_TILE_WIDTH_M))
    return index_x, index_y


def terrain_height_at(task: str, x_m: float, y_m: float, *, rough_level: int | None = None) -> float:
    """Top contact height below ``(x, y)`` for reset and diagnostics."""
    validate_terrain_task(task)
    if task == "flat":
        return 0.0
    if task == "slope_up":
        return float(np.clip((x_m - SLOPE_START_X_M) * math.tan(SLOPE_ANGLE_RAD), 0.0, SLOPE_LENGTH_M * math.tan(SLOPE_ANGLE_RAD)))
    if task == "slope_down":
        return float(np.clip((SLOPE_START_X_M + SLOPE_LENGTH_M - x_m) * math.tan(SLOPE_ANGLE_RAD), 0.0, SLOPE_LENGTH_M * math.tan(SLOPE_ANGLE_RAD)))
    level = validate_rough_level(rough_level)
    if not (
        ROUGH_START_X_M <= x_m <= ROUGH_START_X_M + ROUGH_TILE_COUNT_X * ROUGH_TILE_LENGTH_M
        and ROUGH_START_Y_M <= y_m <= ROUGH_START_Y_M + ROUGH_TILE_COUNT_Y * ROUGH_TILE_WIDTH_M
    ):
        return 0.0
    return _rough_height_at(x_m, y_m, level)


def terrain_asset_xml(task: str) -> str:
    """Return MuJoCo assets required by a terrain task."""
    validate_terrain_task(task)
    if task != "rough":
        return ""
    elevation = " ".join(
        f"{0.5 + 0.5 * _rough_pattern(index_x * (ROUGH_TILE_COUNT_X - 1) / (ROUGH_HFIELD_SAMPLES_X - 1), index_y * (ROUGH_TILE_COUNT_Y - 1) / (ROUGH_HFIELD_SAMPLES_Y - 1)):.7f}"
        for index_y in range(ROUGH_HFIELD_SAMPLES_Y)
        for index_x in range(ROUGH_HFIELD_SAMPLES_X)
    )
    return (
        '<texture name="rough_terrain_texture" type="2d" builtin="flat" mark="random" '
        'rgb1=".19 .40 .095" markrgb=".095 .235 .040" random=".085" width="512" height="512"/>'
        '<material name="rough_terrain_material" texture="rough_terrain_texture" '
        'texrepeat="3 1" texuniform="true" reflectance=".10" specular=".18" shininess=".42"/>'
        f'<hfield name="{ROUGH_HFIELD_NAME}" nrow="{ROUGH_HFIELD_SAMPLES_Y}" '
        f'ncol="{ROUGH_HFIELD_SAMPLES_X}" '
        f'size="{0.5 * ROUGH_TILE_COUNT_X * ROUGH_TILE_LENGTH_M:.6f} '
        f'{0.5 * ROUGH_TILE_COUNT_Y * ROUGH_TILE_WIDTH_M:.6f} '
        f'{2.0 * ROUGH_LEVEL_AMPLITUDE_M[3]:.6f} {ROUGH_TILE_HALF_THICKNESS_M:.6f}" '
        f'elevation="{elevation}"/>'
    )


def terrain_xml(task: str) -> str:
    """Return visible collision geometry for one terrain task."""
    validate_terrain_task(task)
    if task == "flat":
        return ""
    if task in ("slope_up", "slope_down"):
        sign = 1.0 if task == "slope_up" else -1.0
        # See terrain_height_at(): set the top face to the exact requested
        # line z(x), including the rotated box's finite thickness.
        center_x = SLOPE_START_X_M + 0.5 * SLOPE_LENGTH_M
        theta = -sign * SLOPE_ANGLE_RAD
        if task == "slope_up":
            center_z = math.tan(SLOPE_ANGLE_RAD) * (center_x - SLOPE_START_X_M) - SLOPE_HALF_THICKNESS_M / math.cos(SLOPE_ANGLE_RAD)
        else:
            top_at_start = SLOPE_LENGTH_M * math.tan(SLOPE_ANGLE_RAD)
            center_z = top_at_start + math.tan(SLOPE_ANGLE_RAD) * SLOPE_START_X_M - math.tan(SLOPE_ANGLE_RAD) * center_x - SLOPE_HALF_THICKNESS_M / math.cos(SLOPE_ANGLE_RAD)
        colour = ".22 .43 .25 1" if task == "slope_up" else ".28 .34 .19 1"
        return (
            f'<geom name="terrain_{task}" type="box" '
            f'pos="{center_x:.6f} 0 {center_z:.6f}" euler="0 {theta:.8f} 0" '
            f'size="{0.5 * SLOPE_LENGTH_M:.4f} {SLOPE_HALF_WIDTH_M:.4f} {SLOPE_HALF_THICKNESS_M:.4f}" '
            f'rgba="{colour}" friction="1.10 .020 .010" condim="3" contype="1" conaffinity="1"/>'
        )
    center_x = ROUGH_START_X_M + 0.5 * ROUGH_TILE_COUNT_X * ROUGH_TILE_LENGTH_M
    center_y = ROUGH_START_Y_M + 0.5 * ROUGH_TILE_COUNT_Y * ROUGH_TILE_WIDTH_M
    # hfield elevation is normalized.  Its world Z origin is the requested
    # base height minus the maximum amplitude, so level 3 spans ±80 mm.
    return (
        f'<geom name="{ROUGH_HFIELD_GEOM_NAME}" type="hfield" hfield="{ROUGH_HFIELD_NAME}" '
        f'pos="{center_x:.6f} {center_y:.6f} {ROUGH_BASE_HEIGHT_M - ROUGH_LEVEL_AMPLITUDE_M[3]:.6f}" '
        f'material="rough_terrain_material" rgba=".82 .82 .82 1" '
        f'friction="1.10 .020 .010" condim="3" contype="1" conaffinity="1"/>'
    )


def terrain_geom_names(task: str) -> tuple[str, ...]:
    validate_terrain_task(task)
    if task in ("slope_up", "slope_down"):
        return (f"terrain_{task}",)
    if task == "rough":
        return (ROUGH_HFIELD_GEOM_NAME,)
    return ()


def configure_rough_terrain(model: mujoco.MjModel, *, level: int) -> None:
    """Set all rough tiles to one real collision severity before an episode."""
    level = validate_rough_level(level)
    hfield_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, ROUGH_HFIELD_NAME)
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, ROUGH_HFIELD_GEOM_NAME)
    if hfield_id < 0 or geom_id < 0:
        raise RuntimeError("rough hfield missing from MuJoCo model")
    start = int(model.hfield_adr[hfield_id])
    count = int(model.hfield_nrow[hfield_id] * model.hfield_ncol[hfield_id])
    scale = ROUGH_LEVEL_AMPLITUDE_M[level] / (2.0 * ROUGH_LEVEL_AMPLITUDE_M[3])
    elevation = np.array(
        [
            0.5 + scale * _rough_pattern(
                index_x * (ROUGH_TILE_COUNT_X - 1) / (ROUGH_HFIELD_SAMPLES_X - 1),
                index_y * (ROUGH_TILE_COUNT_Y - 1) / (ROUGH_HFIELD_SAMPLES_Y - 1),
            )
            for index_y in range(ROUGH_HFIELD_SAMPLES_Y)
            for index_x in range(ROUGH_HFIELD_SAMPLES_X)
        ],
        dtype=np.float64,
    )
    if elevation.size != count:
        raise RuntimeError("rough hfield sample count mismatch")
    model.hfield_data[start:start + count] = elevation
    # The material carries the terrain's granular colour.  Keep the geom
    # multiplier neutral; the old dark level tint multiplied the material and
    # turned an outdoor surface almost black in the third-person recording.
    model.geom_rgba[geom_id] = (1.0, 1.0, 1.0, 1.0)


def terrain_metadata(task: str, level: int | None = None) -> dict[str, float | int | str]:
    """Serializable task description for metrics/dashboard manifests."""
    validate_terrain_task(task)
    if task in ("slope_up", "slope_down"):
        return {
            "task": task,
            "display_name": terrain_display_name(task),
            "slope_grade_percent": SLOPE_GRADE_PERCENT,
            "slope_angle_deg": SLOPE_ANGLE_DEG,
            "route_length_m": SLOPE_LENGTH_M,
            "surface_friction": 1.10,
        }
    if task == "rough":
        level = validate_rough_level(level)
        return {
            "task": task,
            "display_name": terrain_display_name(task, level),
            "rough_level": level,
            "height_amplitude_mm": int(round(1000.0 * ROUGH_LEVEL_AMPLITUDE_M[level])),
            "tile_size_m": ROUGH_TILE_LENGTH_M,
            "surface_friction": 1.10,
        }
    return {"task": "flat", "display_name": "평지", "surface_friction": 0.75}
