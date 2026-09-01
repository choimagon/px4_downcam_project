"""MuJoCo environment: an X500 lands on a QR pad rigidly mounted on Unitree Go2.

The Go2 MJCF and visual meshes come directly from Unitree's official
``unitree_mujoco`` repository.  The default gait follows the official
``unitree_rl_mjlab`` Go2 flat velocity-task configuration.  An optional PPO
adapter trained from ``yang-zj1026/legged-loco``'s Go2 task contract can
replace that low-level joint residual at inference time.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .go2_terrain import (
    TERRAIN_SPEED_MULTIPLIER,
    TERRAIN_TASKS,
    configure_rough_terrain,
    terrain_geom_names,
    terrain_asset_xml,
    terrain_height_at,
    terrain_initial_pitch_rad,
    terrain_metadata,
    terrain_xml,
    validate_rough_level,
    validate_terrain_task,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GO2_XML_SOURCE = PROJECT_ROOT / "third_party" / "unitree_mujoco" / "unitree_robots" / "go2" / "go2.xml"
GO2_MESH_DIR = GO2_XML_SOURCE.parent / "assets"
X500_MESH_DIR = PROJECT_ROOT / "assets" / "mujoco_x500"
GO2_STAND_POSE = np.tile(np.array([0.0, 0.805, -1.610], dtype=np.float64), 4)
GO2_STAND_POSE[0::3] = np.array([0.10, -0.10, 0.10, -0.10], dtype=np.float64)

# This tuple is the auditable policy-input contract.  Every element is
# available from the stock X500/PX4 stack: a QR detector/PnP pipeline on the
# downward camera, QR centre rate, or the onboard state estimator.
# Go2/base/pad state is intentionally absent.
DRONE_OBSERVATION_NAMES = (
    "qr_center_u",
    "qr_center_v",
    "qr_pnp_depth",
    "qr_detected",
    "qr_center_rate_u",
    "qr_center_rate_v",
    "drone_vertical_velocity",
)

# Stock PX4 x500_mono_cam_down camera contract (1280x960 @ 30 Hz).
CAMERA_WIDTH_PX = 1280
CAMERA_HEIGHT_PX = 960
CAMERA_HFOV_RAD = 1.74
CAMERA_VFOV_RAD = 2.0 * math.atan(math.tan(CAMERA_HFOV_RAD / 2.0) / (CAMERA_WIDTH_PX / CAMERA_HEIGHT_PX))
CAMERA_FRAME_PERIOD_S = 1.0 / 30.0
CAMERA_NEAR_M = 0.10
QR_SIZE_M = 0.23
QR_MIN_DETECT_PX = 20.0
CAMERA_POSITION_BODY = np.array([0.0, 0.0, -0.065], dtype=np.float64)
# The stock camera declaration has no additional quaternion, so its calibrated
# axes are fixed to the X500 body axes.  This is a rigid extrinsic, not target
# or simulator-world state.
CAMERA_ROTATION_BODY = np.eye(3, dtype=np.float64)
CAMERA_PNP_ROTATION_NOISE_BASE_DEG = 0.15
CAMERA_PNP_ROTATION_NOISE_PER_M_DEG = 0.03
ESTIMATOR_PERIOD_S = 0.02
QR_LANDING_SURFACE_HALF_THICKNESS_M = 0.004
# The visible QR ink sits three micrometres above the physical board top.  This
# is solely the thickness of an ink/render layer: it prevents coplanar
# z-fighting in the down-camera image while keeping the colliding board at the
# QR floor to a sub-pixel, physically negligible tolerance.
QR_LANDING_SURFACE_TOP_M = 0.0764
QR_INK_RENDER_CLEARANCE_M = 0.000003
QR_PRINT_TOP_M = QR_LANDING_SURFACE_TOP_M + QR_INK_RENDER_CLEARANCE_M
QR_LANDING_SURFACE_CENTER_Z_M = QR_LANDING_SURFACE_TOP_M - QR_LANDING_SURFACE_HALF_THICKNESS_M
QR_CENTER_SITE_Z_M = 0.078
# The imported stock X500 frame mesh is rendered 25 mm above the body origin.
# Its two skid rails reach mesh Z=-0.25259951 m, so the visible sole is at
# -0.22759951 m in the drone body frame.  Contact geoms must share this plane;
# the previous -0.126 m collision sole floated 101.6 mm above the rendered
# skids and made the stock legs visibly pass through the QR deck.
X500_VISUAL_SKID_BOTTOM_BODY_Z_M = 0.025 - 0.25259951
X500_SKID_HALF_SIZE_M = (0.125, 0.0075, 0.0075)
X500_SKID_LATERAL_OFFSET_M = 0.132
X500_SKID_CENTER_BODY_Z_M = (
    X500_VISUAL_SKID_BOTTOM_BODY_Z_M + X500_SKID_HALF_SIZE_M[2]
)
# qr_center is 1.6 mm above the physical QR board top.  The decorative ink
# adds only 3 μm above that plane to avoid renderer z-fighting.
X500_NOMINAL_TOUCHDOWN_RELATIVE_HEIGHT_M = (
    QR_LANDING_SURFACE_TOP_M - QR_CENTER_SITE_Z_M - X500_VISUAL_SKID_BOTTOM_BODY_Z_M
)
FINAL_APPROACH_START_HEIGHT_M = 0.34
FINAL_PRECISION_DESCENT_HEIGHT_M = 0.30
FINAL_PRECISION_TARGET_HEIGHT_M = X500_NOMINAL_TOUCHDOWN_RELATIVE_HEIGHT_M
SUCCESS_MAX_RELATIVE_HEIGHT_M = 0.245
OFF_CENTRE_HARD_LANDING_HEIGHT_M = 0.220
DEEP_PENETRATION_HEIGHT_M = 0.190
SEARCH_AREA_CENTER_WORLD = np.array([0.0, 0.0], dtype=np.float64)
SEARCH_ALTITUDE_WORLD_M = 2.72
FINAL_APPROACH_MEMORY_S = 2.00
TRACKING_MEMORY_S = 0.40
# Learned policies may only trim the high-authority visual servo.  A 0.1 cm/s
# residual cannot hold the aircraft outside the 4 cm alignment gate even if a
# replay-based actor saturates, while it still permits algorithm-specific
# fine correction during approach.
LANDING_POLICY_RESIDUAL_SPEED_MPS = 0.001
LANDING_POLICY_TRAINING_RESIDUAL_SPEED_MPS = 0.002
LANDING_POLICY_RESIDUAL_CUTOFF_HEIGHT_M = 0.45
LANDING_POLICY_RESIDUAL_FULL_HEIGHT_M = 1.20
FINAL_BLIND_DESCENT_SPEED_MPS = 0.16
FINAL_RETRY_BLIND_DESCENT_SPEED_MPS = 0.14
FINAL_APPROACH_ALIGNMENT_M = 0.030
IMU_IMPACT_DELTA_MPS2 = 4.0
IMU_SETTLE_TIME_S = 0.35
# Once the stock IMU detects touchdown, reduce collective instead of trying
# to arrest the last few cm/s with extra upward thrust.  The old velocity
# loop could combine an above-hover command with the deck normal impulse,
# bounce the leading skid pair clear, and trigger repeated hard-profile
# relandings.  A short 88% hover-thrust settle is the contact-sensor-free
# equivalent of a real multirotor touchdown thrust cut.
IMU_SETTLE_THRUST_FRACTION = 0.88
LANDING_RETRY_CLIMB_SPEED_MPS = 0.45
LANDING_RETRY_REACQUIRE_DEPTH_M = 0.30
IMU_IMPACT_MAX_VISUAL_HEIGHT_M = 0.25


# Evaluation difficulties share the identical 2--7 m aerial start, noise,
# wind and touchdown tolerance.  Only Go2 speed and route-turn complexity
# change, so beginner/intermediate/advanced results remain comparable.
GO2_PROFILES: dict[str, dict[str, Any]] = {
    "train": {"max_steps": 900, "radius": (2.01, 6.90), "altitude": (1.20, 1.80), "path_speed": 0.95, "turn_angle_rad": 0.36, "turn_frequency_hz": 0.12, "speed_modulation": 0.12, "wind": 0.025, "dropout": 0.010, "alignment": 0.040, "landing": 0.055},
    "easy": {"max_steps": 850, "radius": (2.01, 6.90), "altitude": (1.20, 1.80), "path_speed": 0.70, "turn_angle_rad": 0.10, "turn_frequency_hz": 0.05, "speed_modulation": 0.04, "wind": 0.025, "dropout": 0.010, "alignment": 0.040, "landing": 0.055},
    "medium": {"max_steps": 1_000, "radius": (2.01, 6.90), "altitude": (1.20, 1.80), "path_speed": 0.90, "turn_angle_rad": 0.28, "turn_frequency_hz": 0.09, "speed_modulation": 0.10, "wind": 0.025, "dropout": 0.010, "alignment": 0.040, "landing": 0.055},
    "hard": {"max_steps": 1_100, "radius": (2.01, 6.90), "altitude": (1.20, 1.80), "path_speed": 1.10, "turn_angle_rad": 0.48, "turn_frequency_hz": 0.14, "speed_modulation": 0.16, "wind": 0.025, "dropout": 0.010, "alignment": 0.040, "landing": 0.055},
}


def _landing_contact_xml(terrain_task: str) -> str:
    """Return matched skid/deck contact parameters for the current course."""
    # The tiled 80 mm course transmits sharp deck accelerations at touchdown,
    # so it needs direct stiffness to keep the visible rail above the visible
    # board.  A 10% moving plane instead needs the original compliant pair so
    # both continuous rails can settle together rather than rebound on a
    # single downhill edge.
    if terrain_task == "rough":
        return 'solref="-100000 -100" solimp=".99 .999 .0001"'
    return 'solref=".008 1" solimp=".96 .99 .001"'


def _go2_mount_xml(terrain_task: str) -> str:
    """A low-profile, rigid dorsal bridge that clears all Go2 leg linkages."""
    landing_contact = _landing_contact_xml(terrain_task)
    return f"""
      <!-- Rigid 0.22 kg dorsal bridge.  It is a child of base_link (no joint),
           so it cannot slide or wobble independently of Go2.  Two narrow
           rails run along the body above the hip sweep volume.  A 23 cm
           visual QR is printed on the same top plane as a reduced 36 cm
           physical deck.  The X500's two continuous stock landing soles are
           explicit visible MuJoCo collision objects, 25.0 cm long and
           centred 26.4 cm apart, so both contact the visible QR plate itself
           rather than a hidden surface. -->
      <body name="qr_mount" pos="-0.015 0 0.055">
        <inertial pos="0 0 .046" mass="0.22" diaginertia="0.0032 0.0032 0.0044"/>
        <geom name="mount_rail_left" type="box" pos="0 .115 .026" size=".165 .010 .016" rgba=".20 .25 .30 1" contype="0" conaffinity="0"/>
        <geom name="mount_rail_right" type="box" pos="0 -.115 .026" size=".165 .010 .016" rgba=".20 .25 .30 1" contype="0" conaffinity="0"/>
        <geom name="mount_cross_front" type="box" pos=".130 0 .038" size=".010 .125 .012" rgba=".32 .38 .44 1" contype="0" conaffinity="0"/>
        <geom name="mount_cross_rear" type="box" pos="-.130 0 .038" size=".010 .125 .012" rgba=".32 .38 .44 1" contype="0" conaffinity="0"/>
        <!-- This is the visible 36 cm QR deck.  Its top face is exactly the
             contact face below; the two geoms share dimensions and a top
             plane so the visible board is not offset from the physics. -->
        <geom name="qr_deck" type="box" pos="0 0 {QR_LANDING_SURFACE_TOP_M - 0.010:.4f}" size=".180 .180 .010" rgba=".08 .12 .18 1" contype="0" conaffinity="0"/>
        <!-- This is the 36 cm physical QR board.  It is transparent only to
             avoid drawing twice over qr_deck; it has the same top plane and
             all collision/contact parameters. -->
        <geom name="landing_surface" type="box" pos="0 0 {QR_LANDING_SURFACE_CENTER_Z_M:.4f}" size=".180 .180 {QR_LANDING_SURFACE_HALF_THICKNESS_M:.4f}" rgba=".13 .19 .27 0" contype="1" conaffinity="1" condim="3" friction=".95 .015 .001" {landing_contact}/>
        <!-- The ink is a three-micrometre visual layer over the physical board
             so neither the white base nor black modules have coplanar pixels
             that could flicker in the camera renderer. -->
        <geom name="qr_print_base" type="box" pos="0 0 {QR_LANDING_SURFACE_TOP_M + 0.000001:.6f}" size=".115 .115 .000001" rgba=".98 .98 .96 1" contype="0" conaffinity="0"/>
        <geom name="qr_black_nw" type="box" pos="-.066 .066 {QR_LANDING_SURFACE_TOP_M + 0.0000025:.7f}" size=".0225 .0225 .0000005" rgba=".012 .012 .015 1" contype="0" conaffinity="0"/>
        <geom name="qr_black_ne" type="box" pos=" .066 .066 {QR_LANDING_SURFACE_TOP_M + 0.0000025:.7f}" size=".0225 .0225 .0000005" rgba=".012 .012 .015 1" contype="0" conaffinity="0"/>
        <geom name="qr_black_sw" type="box" pos="-.066 -.066 {QR_LANDING_SURFACE_TOP_M + 0.0000025:.7f}" size=".0225 .0225 .0000005" rgba=".012 .012 .015 1" contype="0" conaffinity="0"/>
        <geom name="qr_black_center" type="box" pos=" .010 -.005 {QR_LANDING_SURFACE_TOP_M + 0.0000025:.7f}" size=".015 .015 .0000005" rgba=".012 .012 .015 1" contype="0" conaffinity="0"/>
        <geom name="qr_black_a" type="box" pos="-.035 -.0325 {QR_LANDING_SURFACE_TOP_M + 0.0000025:.7f}" size=".010 .010 .0000005" rgba=".012 .012 .015 1" contype="0" conaffinity="0"/>
        <geom name="qr_black_b" type="box" pos=" .0525 -.050 {QR_LANDING_SURFACE_TOP_M + 0.0000025:.7f}" size=".009 .009 .0000005" rgba=".012 .012 .015 1" contype="0" conaffinity="0"/>
        <geom name="qr_black_c" type="box" pos=" .000 .0525 {QR_LANDING_SURFACE_TOP_M + 0.0000025:.7f}" size=".009 .009 .0000005" rgba=".012 .012 .015 1" contype="0" conaffinity="0"/>
        <!-- Sensor reference only.  Keep it invisible so no debug sphere
             obscures the printed QR in the recorded down-camera view. -->
        <site name="qr_center" pos="0 0 {QR_CENTER_SITE_Z_M:.4f}" size=".006" rgba="0 0 0 0"/>
      </body>
    """


def _annulus_ring_xml(radius_m: float, rgba: str, *, segments: int = 72) -> str:
    """Return a visual-only segmented floor ring for the 2--7 m spawn band."""
    pieces: list[str] = []
    for index in range(segments):
        angle_a = 2.0 * math.pi * index / segments
        angle_b = 2.0 * math.pi * (index + 1) / segments
        x1, y1 = radius_m * math.cos(angle_a), radius_m * math.sin(angle_a)
        x2, y2 = radius_m * math.cos(angle_b), radius_m * math.sin(angle_b)
        pieces.append(
            f'<geom name="spawn_ring_{radius_m:g}_{index}" type="capsule" '
            f'fromto="{x1:.5f} {y1:.5f} .009 {x2:.5f} {y2:.5f} .009" '
            f'size=".026" rgba="{rgba}" contype="0" conaffinity="0" group="2"/>'
        )
    return "\n".join(pieces)


def _world_xml(terrain_task: str) -> str:
    return f"""
    <light name="key" pos="-4 -4 8" dir=".35 .35 -1" directional="true" diffuse=".95 .95 1"/>
    <light name="fill" pos="6 2 5" dir="-1 -.2 -.8" directional="true" diffuse=".35 .45 .65"/>
    <geom name="ground" type="plane" size="32 32 .1" rgba=".09 .16 .22 1" friction=".75 .020 .001" condim="3"/>
    <geom name="path_lane" type="box" pos="5 0 .001" size="9 .70 .002" rgba=".12 .48 .62 .20" contype="0" conaffinity="0"/>
    {_annulus_ring_xml(2.0, '.15 .65 1.0 .78')}
    {_annulus_ring_xml(7.0, '1.0 .48 .10 .82')}
    {terrain_xml(terrain_task)}
    """


def _drone_xml(terrain_task: str) -> str:
    landing_contact = _landing_contact_xml(terrain_task)
    return f"""
    <body name="drone" pos="0 0 1.4">
      <freejoint name="drone_freejoint"/>
      <inertial pos="0 0 0" mass="2.0" diaginertia=".021666667 .021666667 .040000000"/>
      <geom name="drone_collision" type="box" pos="0 0 .007" size=".1768 .1768 .025" rgba="0 0 0 0" contype="0" conaffinity="0" group="1"/>
      <geom name="x500_frame_visual" type="mesh" mesh="x500_frame" pos="0 0 .025" euler="0 0 3.141593" group="1" contype="0" conaffinity="0"/>
      <!-- Base/motor/bell/rotor transforms below are the PX4 Gazebo x500_base
           SDF transforms.  In particular, each propeller retains its mesh
           local pose (-.022, -.146384615, -.016) below the named rotor axis;
           using the rotor axis directly displaced it by ~148 mm in the
           converted mesh frame. -->
      <geom name="motor_front_left" type="mesh" mesh="x500_motor" pos=" .174  .174 .032" euler="0 0 -.45" material="x500_metal" group="1" contype="0" conaffinity="0"/>
      <geom name="motor_rear_left" type="mesh" mesh="x500_motor" pos="-.174  .174 .032" euler="0 0 -.45" material="x500_metal" group="1" contype="0" conaffinity="0"/>
      <geom name="motor_front_right" type="mesh" mesh="x500_motor" pos=" .174 -.174 .032" euler="0 0 -.45" material="x500_metal" group="1" contype="0" conaffinity="0"/>
      <geom name="motor_rear_right" type="mesh" mesh="x500_motor" pos="-.174 -.174 .032" euler="0 0 -.45" material="x500_metal" group="1" contype="0" conaffinity="0"/>
      <geom name="bell_front_right" type="mesh" mesh="x500_bell" pos=" .174 -.174 .028" material="x500_metal" group="1" contype="0" conaffinity="0"/>
      <geom name="bell_rear_left" type="mesh" mesh="x500_bell" pos="-.174  .174 .028" material="x500_metal" group="1" contype="0" conaffinity="0"/>
      <geom name="bell_front_left" type="mesh" mesh="x500_bell" pos=" .174  .174 .028" material="x500_metal" group="1" contype="0" conaffinity="0"/>
      <geom name="bell_rear_right" type="mesh" mesh="x500_bell" pos="-.174 -.174 .028" material="x500_metal" group="1" contype="0" conaffinity="0"/>
      <site name="rotor_axis_front_right" pos=" .174 -.174 .060" size=".003" rgba="0 0 0 0"/>
      <site name="rotor_axis_rear_left" pos="-.174  .174 .060" size=".003" rgba="0 0 0 0"/>
      <site name="rotor_axis_front_left" pos=" .174  .174 .060" size=".003" rgba="0 0 0 0"/>
      <site name="rotor_axis_rear_right" pos="-.174 -.174 .060" size=".003" rgba="0 0 0 0"/>
      <geom name="propeller_front_right" type="mesh" mesh="x500_prop_ccw" pos=" .152 -.320384615 .044" material="arm" group="1" contype="0" conaffinity="0"/>
      <geom name="propeller_rear_left" type="mesh" mesh="x500_prop_ccw" pos="-.196  .027615385 .044" material="arm" group="1" contype="0" conaffinity="0"/>
      <!-- The CW STL conversion carries a 1.56 mm Y centroid offset relative
           to the source DAE.  The two positions below compensate it so all
           four rendered propeller centres sit on their motor axes. -->
      <geom name="propeller_front_left" type="mesh" mesh="x500_prop_cw" pos=" .152052  .026055 .044" material="arm" group="1" contype="0" conaffinity="0"/>
      <geom name="propeller_rear_right" type="mesh" mesh="x500_prop_cw" pos="-.195948 -.321945 .044" material="arm" group="1" contype="0" conaffinity="0"/>
      <!-- These are explicit, visible MuJoCo landing-gear sole objects.  They
           use the exact PX4 Gazebo X500 250 x 15 x 15 mm rail dimensions,
           including collision, friction, and compliant contact.  Their bottom
           plane equals the imported visual skid sole, so the object that the
           user sees resting on the QR is the object carrying the contact. -->
      <geom name="drone_skid_left" type="box" pos="0  {X500_SKID_LATERAL_OFFSET_M:.3f} {X500_SKID_CENTER_BODY_Z_M:.8f}" size="{X500_SKID_HALF_SIZE_M[0]:.4f} {X500_SKID_HALF_SIZE_M[1]:.4f} {X500_SKID_HALF_SIZE_M[2]:.4f}" rgba=".10 .13 .17 1" group="1" contype="1" conaffinity="1" condim="3" friction=".95 .015 .001" {landing_contact}/>
      <geom name="drone_skid_right" type="box" pos="0 -{X500_SKID_LATERAL_OFFSET_M:.3f} {X500_SKID_CENTER_BODY_Z_M:.8f}" size="{X500_SKID_HALF_SIZE_M[0]:.4f} {X500_SKID_HALF_SIZE_M[1]:.4f} {X500_SKID_HALF_SIZE_M[2]:.4f}" rgba=".10 .13 .17 1" group="1" contype="1" conaffinity="1" condim="3" friction=".95 .015 .001" {landing_contact}/>
      <!-- The actual camera remains at down_camera below.  These two crude
           primitives were render-only housing/lens placeholders, not sensors;
           hide them so no black box floats under the imported stock model. -->
      <geom name="mono_cam_housing" type="box" pos="0 0 -.100" size=".010 .020 .020" rgba="0 0 0 0" group="1" contype="0" conaffinity="0"/>
      <geom name="mono_cam_lens" type="cylinder" pos="0 0 -.123" size=".009 .006" rgba="0 0 0 0" group="1" contype="0" conaffinity="0"/>
      <site name="drone_imu" pos="0 0 .010" size=".004" rgba="0 0 0 0"/>
      <!-- The optical origin is 162.6 mm above the aligned skid sole, so the
           camera stays clear of the QR deck at physical touchdown. -->
      <camera name="down_camera" pos="0 0 -.065" fovy="83.27"/>
    </body>
    """


def build_go2_landing_xml(*, include_drone: bool = True, terrain_task: str = "flat") -> str:
    """Patch the official Go2 MJCF in-memory; no simplified dog substitute.

    ``include_drone=False`` is used by the standalone low-level locomotion
    learner.  It keeps exactly the same official Go2 model and the 0.22 kg
    dorsal QR fixture, while avoiding a free X500 body in that training task.
    """
    terrain_task = validate_terrain_task(terrain_task)
    if not GO2_XML_SOURCE.exists():
        raise FileNotFoundError(f"Official Go2 MJCF not present: {GO2_XML_SOURCE}")
    template = GO2_XML_SOURCE.read_text(encoding="utf-8")
    template = template.replace(
        '<compiler angle="radian" meshdir="assets" autolimits="true" />',
        f'<compiler angle="radian" meshdir="{GO2_MESH_DIR.as_posix()}" autolimits="true" />',
        1,
    )
    template = template.replace(
        '<option cone="elliptic" impratio="100" />',
        '<option timestep="0.005" gravity="0 0 -9.81" integrator="RK4" cone="elliptic" impratio="100"/>\n'
        '  <visual><global offwidth="1280" offheight="720"/><map znear="0.001" zfar="100"/><rgba haze=".14 .22 .32 1"/></visual>',
        1,
    )
    template = template.replace(
        'friction="0.4 0.02 0.01"',
        'friction="1.05 0.020 0.010"',
        1,
    )
    assets = terrain_asset_xml(terrain_task)
    if include_drone:
        assets += f'''
        <material name="x500_metal" rgba=".10 .13 .17 1"/><material name="arm" rgba=".05 .10 .16 1"/>
        <material name="camera_housing" rgba=".035 .045 .060 1"/><material name="lens" rgba=".15 .35 .46 1"/>
        <mesh name="x500_frame" file="{(X500_MESH_DIR / 'x500_frame.obj').as_posix()}"/>
        <mesh name="x500_motor" file="{(X500_MESH_DIR / 'x500_motor.obj').as_posix()}"/>
        <mesh name="x500_bell" file="{(X500_MESH_DIR / 'x500_bell.obj').as_posix()}"/>
        <mesh name="x500_prop_cw" file="{(X500_MESH_DIR / '1345_prop_cw.stl').as_posix()}" scale=".846154 .846154 .846154"/>
        <mesh name="x500_prop_ccw" file="{(X500_MESH_DIR / '1345_prop_ccw.stl').as_posix()}" scale=".846154 .846154 .846154"/>
        '''
    template = template.replace("  </asset>", assets + "\n  </asset>", 1)
    mount_needle = '<site name="imu" pos="-0.02557 0 0.04232" />'
    if mount_needle not in template:
        raise RuntimeError("Could not locate official Go2 base_link mount point")
    template = template.replace(mount_needle, mount_needle + _go2_mount_xml(terrain_task), 1)
    template = template.replace("  <worldbody>", "  <worldbody>" + _world_xml(terrain_task), 1)
    insert_at = template.rfind("    </body>\n  </worldbody>")
    if insert_at < 0:
        raise RuntimeError("Could not locate official Go2 worldbody boundary")
    if include_drone:
        insert_at += len("    </body>")
        template = template[:insert_at] + _drone_xml(terrain_task) + template[insert_at:]
        # Explicit simulated onboard sensors.  The flight controller below
        # reads these sensor channels rather than Go2 body state.
        template = template.replace(
            "  <sensor>",
            '''  <sensor>
    <framepos name="drone_gps_position" objtype="site" objname="drone_imu"/>
    <framelinvel name="drone_gps_velocity" objtype="site" objname="drone_imu"/>
    <framequat name="drone_attitude" objtype="site" objname="drone_imu"/>
    <gyro name="drone_gyro" site="drone_imu"/>
    <accelerometer name="drone_accelerometer" site="drone_imu"/>''',
            1,
        )
    return template


class Go2BackQrLandingEnv(gym.Env[np.ndarray, np.ndarray]):
    """Land an X500 on a moving, physically attached Go2-back QR plate."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        difficulty: str = "train",
        seed: int | None = None,
        dt: float = 0.10,
        locomotion_model: str | Path | None = None,
        terrain_task: str = "flat",
        rough_level: int | None = None,
        policy_residual_speed_mps: float = LANDING_POLICY_RESIDUAL_SPEED_MPS,
    ) -> None:
        super().__init__()
        if difficulty not in GO2_PROFILES:
            raise ValueError(f"difficulty must be one of: {', '.join(GO2_PROFILES)}")
        self.difficulty = difficulty
        self.profile = GO2_PROFILES[difficulty]
        self.terrain_task = validate_terrain_task(terrain_task)
        if self.terrain_task != "rough" and rough_level is not None:
            raise ValueError("rough_level is only valid for terrain_task='rough'")
        self._requested_rough_level = validate_rough_level(rough_level) if rough_level is not None else None
        self._active_rough_level = self._requested_rough_level or 2
        self.policy_residual_speed_mps = float(policy_residual_speed_mps)
        if not 0.0 <= self.policy_residual_speed_mps <= 0.01:
            raise ValueError("policy_residual_speed_mps must be within [0, 0.01]")
        self.model = mujoco.MjModel.from_xml_string(build_go2_landing_xml(terrain_task=self.terrain_task))
        self.data = mujoco.MjData(self.model)
        self.base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.mount_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "qr_mount")
        self.drone_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "drone")
        self.qr_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "qr_center")
        self.down_camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "down_camera")
        self.drone_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "drone_freejoint")
        self.drone_qposadr = int(self.model.jnt_qposadr[self.drone_joint_id])
        self.drone_dofadr = int(self.model.jnt_dofadr[self.drone_joint_id])
        self.drone_mass = float(self.model.body_mass[self.drone_id])
        self._drone_sensor_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            for name in (
                "drone_gps_position", "drone_gps_velocity", "drone_attitude",
                "drone_gyro", "drone_accelerometer",
            )
        }
        self.skid_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
                for name in ("drone_skid_left", "drone_skid_right")
            ],
            dtype=np.int32,
        )
        self.landing_surface_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "landing_surface")
        self.go2_joint_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in (
                    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
                    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint", "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
                )
            ],
            dtype=np.int32,
        )
        self.go2_qposadr = self.model.jnt_qposadr[self.go2_joint_ids].astype(np.int32)
        self.go2_dofadr = self.model.jnt_dofadr[self.go2_joint_ids].astype(np.int32)
        self.go2_actuator_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in ("FR_hip", "FR_thigh", "FR_calf", "FL_hip", "FL_thigh", "FL_calf", "RR_hip", "RR_thigh", "RR_calf", "RL_hip", "RL_thigh", "RL_calf")],
            dtype=np.int32,
        )
        self.go2_foot_geom_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in ("FR", "FL", "RR", "RL")],
            dtype=np.int32,
        )
        self.ground_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        self.terrain_geom_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in terrain_geom_names(self.terrain_task)],
            dtype=np.int32,
        )
        if np.any(self.terrain_geom_ids < 0):
            raise RuntimeError("terrain geometry was not compiled into the landing scene")
        self.physics_steps = max(1, round(dt / self.model.opt.timestep))
        self.dt = self.physics_steps * float(self.model.opt.timestep)
        self.max_steps = int(self.profile["max_steps"])
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self._step_count = 0
        self._aligned_streak = 0
        self._landing_committed = False
        self._approach_recovery = False
        self._dropout = False
        # PX4-estimator surrogate: 50 Hz sample-and-hold of stock GNSS/IMU/
        # barometer-derived state, with small repeatable measurement noise.
        self._estimated_position = np.zeros(3, dtype=np.float64)
        self._estimated_velocity = np.zeros(3, dtype=np.float64)
        self._estimated_rotation = np.eye(3, dtype=np.float64)
        self._estimated_angular_velocity = np.zeros(3, dtype=np.float64)
        self._next_estimator_time = 0.0
        self._next_camera_time = 0.0
        self._last_qr_seen_time = float("-inf")
        self._final_approach = False
        self._final_approach_started = float("-inf")
        self._search_origin_xy = np.zeros(2, dtype=np.float64)
        self._search_altitude = 0.0
        self._search_started = 0.0
        self._wind_force_world = np.zeros(3, dtype=np.float64)
        self._prev_pad_position = np.zeros(3, dtype=np.float64)
        self._pad_velocity = np.zeros(2, dtype=np.float64)
        self._pad_vertical_velocity = 0.0
        # Cached output of the downward-camera QR detector.  Only
        # _update_qr_camera_measurement() may translate simulator geometry
        # into this camera/PnP measurement; policy and control use the cache.
        self._qr_detected = False
        self._qr_center_norm = np.zeros(2, dtype=np.float64)
        self._qr_translation_body = np.zeros(3, dtype=np.float64)
        self._qr_rotation_body = np.eye(3, dtype=np.float64)
        self._qr_center_rate = np.zeros(2, dtype=np.float64)
        self._qr_relative_velocity_world = np.zeros(3, dtype=np.float64)
        self._qr_target_velocity_world = np.zeros(3, dtype=np.float64)
        self._qr_target_position_world = np.zeros(3, dtype=np.float64)
        self._qr_target_rotation_world = np.eye(3, dtype=np.float64)
        self._previous_qr_center = np.zeros(2, dtype=np.float64)
        self._previous_qr_relative_world = np.zeros(3, dtype=np.float64)
        self._previous_qr_measurement_time = 0.0
        self._previous_qr_valid = False
        self._qr_depth = 0.0
        # MuJoCo-only contact mechanics for reward labels, termination and
        # offline evaluation.  Stock X500 has no landing-leg contact/load
        # sensor, so these values never enter observation or _drone_control().
        self._offline_sim_landing_normal_force = 0.0
        self._offline_sim_landing_skid_contact_count = 0
        self._offline_sim_max_contact_penetration = 0.0
        self._touchdown_success_evidence = False
        self._imu_impact_latched = False
        self._imu_impact_time = float("-inf")
        self._commanded_specific_force_body_z = abs(float(self.model.opt.gravity[2]))
        self._landing_retry_active = False
        self._landing_retry_count = 0
        self._path_length = 0.0
        self._previous_base_position = np.zeros(3, dtype=np.float64)
        self._previous_go2_foot_positions = np.zeros((4, 3), dtype=np.float64)
        self._previous_go2_contact_mask = np.zeros(4, dtype=bool)
        self._go2_stance_slip_mps = 0.0
        # Inference can install a render-only observer to capture true 5 ms
        # substeps at 30 fps.  Training and evaluation leave it unset.
        self.physics_observer: Callable[["Go2BackQrLandingEnv"], None] | None = None
        self._legged_loco = None
        if locomotion_model is not None:
            from .go2_legged_loco_adapter import Go2LeggedLocoAdapter

            # Terrain residuals are scaled exactly once in
            # _apply_go2_locomotion(), matching Go2LeggedLocoEnv.  The bridge
            # therefore forwards the policy output unchanged; applying 25%
            # here as well would silently reduce the trained authority to
            # 6.25% in the recorded physical scene.
            terrain_deployment_gain = 1.0 if self.terrain_task != "flat" else 0.50
            self._legged_loco = Go2LeggedLocoAdapter(
                locomotion_model, deployment_action_gain=terrain_deployment_gain
            )
        self.reset(seed=seed)

    @property
    def drone_position(self) -> np.ndarray:
        return self.data.xpos[self.drone_id].copy()

    @property
    def drone_velocity(self) -> np.ndarray:
        return self.data.qvel[self.drone_dofadr:self.drone_dofadr + 3].copy()

    @property
    def pad_position(self) -> np.ndarray:
        return self.data.site_xpos[self.qr_site_id].copy()

    @property
    def base_position(self) -> np.ndarray:
        return self.data.xpos[self.base_id].copy()

    def _sensor(self, name: str) -> np.ndarray:
        """Read one explicit MuJoCo onboard-sensor channel."""
        sensor_id = self._drone_sensor_ids[name]
        address = int(self.model.sensor_adr[sensor_id])
        dimension = int(self.model.sensor_dim[sensor_id])
        return self.data.sensordata[address:address + dimension].copy()

    def _onboard_position(self) -> np.ndarray:
        return self._estimated_position.copy()

    def _onboard_rotation(self) -> np.ndarray:
        return self._estimated_rotation.copy()

    def _onboard_velocity(self) -> np.ndarray:
        return self._estimated_velocity.copy()

    def _onboard_angular_velocity(self) -> np.ndarray:
        return self._estimated_angular_velocity.copy()

    def _update_onboard_estimator(self, *, force: bool = False) -> None:
        """Update a 50 Hz sample-and-hold surrogate for PX4 EKF2 outputs.

        MuJoCo's explicit GNSS/IMU channels are the sensor boundary here.
        Small seeded errors model the fact that real ``vehicle_local_position``
        and attitude estimates are neither noiseless nor available at 200 Hz.
        No Go2, QR-mount, landing-surface or contact state is read.
        """
        now = float(self.data.time)
        if not force and now + 1.0e-9 < self._next_estimator_time:
            return
        raw_position = self._sensor("drone_gps_position")
        raw_velocity = self._sensor("drone_gps_velocity")
        matrix = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(matrix, self._sensor("drone_attitude"))
        raw_rotation = matrix.reshape(3, 3)
        raw_gyro_body = self._sensor("drone_gyro")
        self._estimated_position[:] = raw_position + self.np_random.normal(
            0.0, np.array([0.012, 0.012, 0.018]), size=3
        )
        self._estimated_velocity[:] = raw_velocity + self.np_random.normal(
            0.0, np.array([0.010, 0.010, 0.014]), size=3
        )
        self._estimated_rotation[:] = raw_rotation
        gyro_body = raw_gyro_body + self.np_random.normal(0.0, 0.0015, size=3)
        self._estimated_angular_velocity[:] = raw_rotation @ gyro_body
        self._next_estimator_time = now + ESTIMATOR_PERIOD_S

    def _relative_altitude(self) -> float:
        return max(0.0, float(self.drone_position[2] - self.pad_position[2]))

    def _horizontal_error(self) -> np.ndarray:
        return self.drone_position[:2] - self.pad_position[:2]

    def _path_heading(self, time_s: float) -> float:
        """Smooth world-frame route heading used by the physical gait only."""
        if self.terrain_task != "flat":
            # Terrain PPO certification and its 3x-speed recordings use the
            # same straight forward course as training.  Turning a 10% grade
            # into a separate lateral/yaw task would change the requested
            # speed experiment and excite an unrelated foothold failure.
            return 0.0
        terrain_turn_scale = 1.0 if self.terrain_task == "flat" else 0.24
        turn_angle = terrain_turn_scale * float(self.profile["turn_angle_rad"])
        frequency = float(self.profile["turn_frequency_hz"])
        phase = 2.0 * math.pi * frequency * time_s
        return turn_angle * (math.sin(phase) + 0.22 * math.sin(0.5 * phase + 0.35))

    def _path_heading_rate(self, time_s: float) -> float:
        if self.terrain_task != "flat":
            return 0.0
        terrain_turn_scale = 1.0 if self.terrain_task == "flat" else 0.24
        turn_angle = terrain_turn_scale * float(self.profile["turn_angle_rad"])
        omega = 2.0 * math.pi * float(self.profile["turn_frequency_hz"])
        phase = omega * time_s
        return turn_angle * omega * (math.cos(phase) + 0.11 * math.cos(0.5 * phase + 0.35))

    def _path_command(self, time_s: float) -> np.ndarray:
        # Match the terrain PPO's requested speed range exactly: three times
        # the former terrain route speed, with grade/footholds still supplied
        # by actual MuJoCo contact rather than a scripted Go2 translation.
        terrain_speed_scale = 1.0 if self.terrain_task == "flat" else (
            TERRAIN_SPEED_MULTIPLIER * (0.52 if self.terrain_task.startswith("slope") else 0.48)
        )
        speed = terrain_speed_scale * float(self.profile["path_speed"])
        speed_modulation = 0.0 if self.terrain_task != "flat" else float(self.profile["speed_modulation"])
        phase = 2.0 * math.pi * float(self.profile["turn_frequency_hz"]) * time_s
        heading = self._path_heading(time_s)
        forward = speed * (1.0 - speed_modulation * 0.5 * (1.0 - math.cos(0.37 * phase)))
        return np.array([forward * math.cos(heading), forward * math.sin(heading), 0.0], dtype=np.float64)

    def _locomotion_command(self, time_s: float) -> np.ndarray:
        """Convert the world route into Go2 body-frame velocity and yaw rate."""
        world_velocity = self._path_command(time_s)[:2]
        if self.terrain_task != "flat":
            # Keep the certified terrain-controller command in the Go2 body
            # frame.  Approach the visible finite-course endpoint at a
            # controlled stop: a longer landing video must not drive an
            # otherwise stable robot off the collision surface after it has
            # already completed the 12 m traversal.
            progress = float(self.base_position[0])
            if progress >= 11.5:
                return np.zeros(3, dtype=np.float64)
            endpoint_scale = float(np.clip((11.5 - progress) / 1.0, 0.0, 1.0))
            return np.array([endpoint_scale * float(world_velocity[0]), 0.0, 0.0], dtype=np.float64)
        rotation = self.data.xmat[self.base_id].reshape(3, 3)
        body_velocity = rotation[:2, :2].T @ world_velocity
        current_yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
        desired_yaw = math.atan2(float(world_velocity[1]), float(world_velocity[0]))
        yaw_error = math.atan2(math.sin(desired_yaw - current_yaw), math.cos(desired_yaw - current_yaw))
        yaw_rate = np.clip(self._path_heading_rate(time_s) + 2.2 * yaw_error, -1.20, 1.20)
        return np.array([body_velocity[0], body_velocity[1], yaw_rate], dtype=np.float64)

    def _update_qr_camera_measurement(self, *, force: bool = False) -> None:
        """Emulate QR corner detection + solvePnP at the downward camera.

        Simulator QR/mount geometry is read *only* in this sensor boundary,
        just as a hardware camera sees photons generated by the real target.
        Downstream policy/control receives only cached marker translation,
        orientation, image center, centre rate and detection validity.  The
        cache updates at the stock camera's 30 Hz rate and holds between
        frames, including the real camera's 0.10 m near plane.
        """
        now = float(self.data.time)
        if not force and now + 1.0e-9 < self._next_camera_time:
            return
        self._next_camera_time = now + CAMERA_FRAME_PERIOD_S

        drone_rotation = self._onboard_rotation()
        drone_position_estimate = self._onboard_position()
        own_velocity = self._onboard_velocity()
        marker_position = self.data.site_xpos[self.qr_site_id].copy()
        marker_rotation = self.data.xmat[self.mount_id].reshape(3, 3).copy()
        camera_position = self.data.cam_xpos[self.down_camera_id].copy()
        camera_rotation = self.data.cam_xmat[self.down_camera_id].reshape(3, 3).copy()
        relative_camera_true = camera_rotation.T @ (marker_position - camera_position)
        depth = max(0.0, -float(relative_camera_true[2]))
        half_width = max(1.0e-6, depth * math.tan(CAMERA_HFOV_RAD / 2.0))
        half_height = max(1.0e-6, depth * math.tan(CAMERA_VFOV_RAD / 2.0))
        center_true = np.array(
            [relative_camera_true[0] / half_width, relative_camera_true[1] / half_height],
            dtype=np.float64,
        )
        marker_width_px = CAMERA_WIDTH_PX * QR_SIZE_M / max(1.0e-6, 2.0 * half_width)
        marker_height_px = CAMERA_HEIGHT_PX * QR_SIZE_M / max(1.0e-6, 2.0 * half_height)
        marker_pixels = min(marker_width_px, marker_height_px)
        marker_facing_camera = float(marker_rotation[:, 2] @ (camera_position - marker_position)) > 0.0
        visible = (
            CAMERA_NEAR_M <= depth < 12.0
            and float(np.max(np.abs(center_true))) <= 1.0
            and marker_pixels >= QR_MIN_DETECT_PX
            and marker_facing_camera
            and not self._dropout
        )

        if not visible:
            self._qr_detected = False
            self._qr_center_norm[:] = 0.0
            self._qr_center_rate[:] = 0.0
            self._qr_depth = 0.0
            self._previous_qr_valid = False
            return

        # Small seeded PnP noise makes the simulation closer to the real
        # detector while preserving repeatable training and evaluation.
        translation_noise = self.np_random.normal(0.0, 0.0015 + 0.0004 * depth, size=3)
        measured_camera_translation = relative_camera_true + translation_noise
        measured_depth = max(0.0, -float(measured_camera_translation[2]))
        measured_half_width = max(1.0e-6, measured_depth * math.tan(CAMERA_HFOV_RAD / 2.0))
        measured_half_height = max(1.0e-6, measured_depth * math.tan(CAMERA_VFOV_RAD / 2.0))
        measured_center = np.clip(
            np.array(
                [
                    measured_camera_translation[0] / measured_half_width,
                    measured_camera_translation[1] / measured_half_height,
                ],
                dtype=np.float64,
            ) + self.np_random.normal(0.0, 0.001, size=2),
            -1.0,
            1.0,
        )
        # solvePnP returns marker orientation relative to the camera.  Keep
        # that same hardware boundary here: perturb the camera-frame pose,
        # apply only the calibrated camera-to-body extrinsic, then reconstruct
        # the world reference with the X500's own attitude estimate.  Never
        # copy the exact MuJoCo marker/world rotation into the controller.
        true_camera_marker_rotation = camera_rotation.T @ marker_rotation
        rotation_noise_sigma = math.radians(
            CAMERA_PNP_ROTATION_NOISE_BASE_DEG
            + CAMERA_PNP_ROTATION_NOISE_PER_M_DEG * depth
        )
        rotation_vector = np.clip(
            self.np_random.normal(0.0, rotation_noise_sigma, size=3),
            -3.0 * rotation_noise_sigma,
            3.0 * rotation_noise_sigma,
        )
        rotation_angle = float(np.linalg.norm(rotation_vector))
        if rotation_angle > 1.0e-12:
            axis = rotation_vector / rotation_angle
            skew = np.array(
                [
                    [0.0, -axis[2], axis[1]],
                    [axis[2], 0.0, -axis[0]],
                    [-axis[1], axis[0], 0.0],
                ],
                dtype=np.float64,
            )
            rotation_noise = (
                np.eye(3, dtype=np.float64)
                + math.sin(rotation_angle) * skew
                + (1.0 - math.cos(rotation_angle)) * (skew @ skew)
            )
        else:
            rotation_noise = np.eye(3, dtype=np.float64)
        measured_camera_rotation = true_camera_marker_rotation @ rotation_noise
        measured_body_rotation = CAMERA_ROTATION_BODY @ measured_camera_rotation
        # Camera axes are fixed to body axes in the stock downward mount.
        # PnP translation is converted through that known extrinsic before
        # the flight controller consumes it.
        measured_body_translation = CAMERA_POSITION_BODY + measured_camera_translation
        relative_world = drone_rotation @ measured_body_translation
        if self._previous_qr_valid:
            measurement_dt = max(1.0e-4, now - self._previous_qr_measurement_time)
            raw_relative_velocity = (relative_world - self._previous_qr_relative_world) / measurement_dt
            raw_target_velocity = np.clip(own_velocity + raw_relative_velocity, -3.0, 3.0)
            raw_center_rate = np.clip((measured_center - self._previous_qr_center) / measurement_dt, -5.0, 5.0)
            self._qr_relative_velocity_world = 0.65 * self._qr_relative_velocity_world + 0.35 * raw_relative_velocity
            self._qr_target_velocity_world = 0.65 * self._qr_target_velocity_world + 0.35 * raw_target_velocity
            self._qr_center_rate = 0.65 * self._qr_center_rate + 0.35 * raw_center_rate
        else:
            # A near-plane/dropout frame marks _previous_qr_valid false.  If
            # the marker flickers back for one frame, preserve the last
            # camera-derived target velocity instead of resetting it to zero;
            # otherwise the drone brakes in the blind gap while the deck keeps
            # walking away.  Long-lost targets still start from zero velocity.
            memory_window = FINAL_APPROACH_MEMORY_S if self._final_approach else TRACKING_MEMORY_S
            if now - self._last_qr_seen_time > memory_window:
                self._qr_relative_velocity_world[:] = 0.0
                self._qr_target_velocity_world[:] = 0.0
            self._qr_center_rate[:] = 0.0
        self._qr_detected = True
        self._qr_center_norm[:] = measured_center
        self._qr_translation_body[:] = measured_body_translation
        self._qr_rotation_body[:] = measured_body_rotation
        self._qr_target_position_world[:] = drone_position_estimate + relative_world
        self._qr_target_rotation_world[:] = drone_rotation @ measured_body_rotation
        self._qr_depth = measured_depth
        self._last_qr_seen_time = now
        self._previous_qr_center[:] = measured_center
        self._previous_qr_relative_world[:] = relative_world
        self._previous_qr_measurement_time = now
        self._previous_qr_valid = True

    def _detected(self, _distance: float | None = None) -> bool:
        """Compatibility accessor backed only by the camera detector cache."""
        return self._qr_detected

    def _observation(self) -> np.ndarray:
        image_center = self._qr_center_norm if self._qr_detected else np.zeros(2, dtype=np.float64)
        center_rate = self._qr_center_rate if self._qr_detected else np.zeros(2, dtype=np.float64)
        pnp_depth = self._qr_depth if self._qr_detected else 0.0
        return np.array(
            [
                image_center[0], image_center[1], min(1.0, pnp_depth / 8.0), float(self._qr_detected),
                np.clip(center_rate[0] / 3.0, -1.0, 1.0), np.clip(center_rate[1] / 3.0, -1.0, 1.0),
                np.clip(self._onboard_velocity()[2] / 3.0, -1.0, 1.0),
            ],
            dtype=np.float32,
        )

    def _search_velocity(self, own_position: np.ndarray) -> np.ndarray:
        """GNSS/barometer-only expanding search around a mission waypoint.

        The fixed world origin is the declared search-area centre, not the
        moving target position.  The spiral therefore remains identical if
        hidden Go2 state is changed; acquisition happens only through QR
        camera frames.
        """
        elapsed = max(0.0, float(self.data.time) - self._search_started)
        # The declared search area is a forward corridor.  Advance along it
        # at a fixed mission-planning speed while sweeping laterally.  This is
        # an open-loop search pattern (time + own EKF pose only), not access to
        # Go2 position, velocity, route phase, or the current QR truth.
        waypoint = SEARCH_AREA_CENTER_WORLD + np.array(
            [1.0 * elapsed, 1.15 * math.sin(0.55 * elapsed)], dtype=np.float64
        )
        error = waypoint - own_position[:2]
        distance = float(np.linalg.norm(error))
        if distance < 1.0e-6:
            return np.zeros(2, dtype=np.float64)
        speed = min(3.60, 2.0 * distance + 0.35)
        return speed * error / distance

    def _go2_targets(self, time_s: float) -> np.ndarray:
        """Trot reference from the official Go2 velocity-task phase offsets."""
        targets = np.tile(np.array([0.0, 0.90, -1.80], dtype=np.float64), 4)
        phase_offsets = (0.0, 0.5, 0.5, 0.0)  # FR, FL, RR, RL from unitree_rl_mjlab
        for leg, offset in enumerate(phase_offsets):
            phase = 2.0 * math.pi * (time_s / 0.60 + offset)
            swing = math.sin(phase)
            lift = max(0.0, swing)
            idx = 3 * leg
            targets[idx] = 0.025 * (1.0 if leg in (0, 2) else -1.0) * math.sin(phase)
            targets[idx + 1] = 0.90 + 0.055 * swing
            targets[idx + 2] = -1.80 - 0.085 * lift + 0.025 * min(0.0, swing)
        return targets

    def _apply_go2_locomotion(self) -> None:
        qpos = self.data.qpos[self.go2_qposadr]
        qvel = self.data.qvel[self.go2_dofadr]
        if self._legged_loco is None and self.terrain_task == "flat":
            targets = self._go2_targets(float(self.data.time))
            kp = np.tile(np.array([20.0, 20.0, 40.0]), 4)
            kd = np.tile(np.array([1.0, 1.0, 2.0]), 4)
        else:
            from .go2_legged_loco_environment import (
                JOINT_RESIDUAL_SCALE_RAD,
                TERRAIN_RESIDUAL_POLICY_GAIN,
                _rpy,
                legged_loco_reference_target,
            )

            residual = (
                self._legged_loco.control(self)
                if self._legged_loco is not None else np.zeros(12, dtype=np.float64)
            )
            time_s = float(self.data.time)
            command = self._locomotion_command(time_s)
            reference = legged_loco_reference_target(
                time_s,
                command,
                fast_terrain_gait=self.terrain_task != "flat",
                terrain_task=self.terrain_task,
                body_rpy=_rpy(self.data.xmat[self.base_id]) if self.terrain_task != "flat" else None,
                body_angular_velocity=self.data.cvel[self.base_id, :3] if self.terrain_task != "flat" else None,
                course_lateral_error_m=float(self.base_position[1]) if self.terrain_task != "flat" else 0.0,
                desired_pitch_rad=terrain_initial_pitch_rad(self.terrain_task),
            )
            next_reference = legged_loco_reference_target(
                time_s + float(self.model.opt.timestep),
                command,
                fast_terrain_gait=self.terrain_task != "flat",
                terrain_task=self.terrain_task,
                body_rpy=_rpy(self.data.xmat[self.base_id]) if self.terrain_task != "flat" else None,
                body_angular_velocity=self.data.cvel[self.base_id, :3] if self.terrain_task != "flat" else None,
                course_lateral_error_m=float(self.base_position[1]) if self.terrain_task != "flat" else 0.0,
                desired_pitch_rad=terrain_initial_pitch_rad(self.terrain_task),
            )
            terrain_residual_gain = TERRAIN_RESIDUAL_POLICY_GAIN if self.terrain_task != "flat" else 1.0
            targets = reference + JOINT_RESIDUAL_SCALE_RAD * terrain_residual_gain * residual
            target_velocity = (next_reference - reference) / float(self.model.opt.timestep)
            # Match legged-loco's Go2 DelayedPDActuatorCfg.
            kp = np.full(12, 60.0)
            kd = np.full(12, 2.0)
        if self._legged_loco is None and self.terrain_task == "flat":
            target_velocity = np.zeros(12, dtype=np.float64)
        torque = kp * (targets - qpos) + kd * (target_velocity - qvel)
        ctrlrange = self.model.actuator_ctrlrange[self.go2_actuator_ids]
        self.data.ctrl[self.go2_actuator_ids] = np.clip(torque, ctrlrange[:, 0], ctrlrange[:, 1])

        # No hidden root wrench.  Go2 motion and balance come only from joint
        # torques transmitted through physical foot contacts.
        self.data.xfrc_applied[self.base_id] = 0.0

    def _contact_calibration(self) -> tuple[int, float, float]:
        contacting_skids: set[int] = set()
        normal_force = 0.0
        penetration = 0.0
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            pair = {int(contact.geom1), int(contact.geom2)}
            matching_skids = [int(skid) for skid in self.skid_ids if int(skid) in pair]
            if self.landing_surface_id not in pair or not matching_skids:
                continue
            contacting_skids.update(matching_skids)
            contact_force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, contact_index, contact_force)
            normal_force += max(0.0, float(contact_force[0]))
            penetration = max(penetration, max(0.0, -float(contact.dist)))
        return len(contacting_skids), normal_force, penetration

    def _go2_ground_contact_mask(self) -> np.ndarray:
        contacts = np.zeros(4, dtype=bool)
        support_geoms = {int(self.ground_geom_id), *(int(geom_id) for geom_id in self.terrain_geom_ids)}
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if not pair.intersection(support_geoms):
                continue
            for foot_index, geom_id in enumerate(self.go2_foot_geom_ids):
                if int(geom_id) in pair:
                    contacts[foot_index] = True
        return contacts

    def _drone_control(self, action: np.ndarray, *, update_alignment: bool) -> None:
        """Control X500 from its onboard sensors and cached QR PnP only.

        Ground-truth Go2/pad position, velocity and attitude are deliberately
        not read here.  Before visual acquisition, a fixed mission-area
        waypoint and an expanding search use only own-position/altitude
        estimates.  They never query the moving target's simulator state.
        """
        action = np.asarray(action, dtype=np.float64).clip(-1.0, 1.0)
        own_position = self._onboard_position()
        own_velocity = self._onboard_velocity()
        drone_rotation = self._onboard_rotation()
        now = float(self.data.time)
        detected = self._qr_detected
        time_since_seen = now - self._last_qr_seen_time
        long_memory_mode = self._final_approach or self._landing_retry_active
        visual_memory_valid = time_since_seen <= (
            FINAL_APPROACH_MEMORY_S if long_memory_mode else TRACKING_MEMORY_S
        )
        final_memory_valid = long_memory_mode and visual_memory_valid
        tracking = detected or visual_memory_valid
        if detected:
            relative_world = drone_rotation @ self._qr_translation_body
        elif final_memory_valid:
            propagated_target = self._qr_target_position_world + max(0.0, time_since_seen) * self._qr_target_velocity_world
            relative_world = propagated_target - own_position
        else:
            relative_world = np.zeros(3, dtype=np.float64)

        if tracking:
            horizontal_error = float(np.linalg.norm(relative_world[:2]))
            relative_height = max(0.0, -float(relative_world[2]))
            # Fast camera-centering approach with PnP-motion feed-forward;
            # the learned action remains a bounded residual.
            # Keep learned exploration useful during acquisition/coarse
            # tracking, then taper it out before precision touchdown.  Even a
            # millimetres-per-second actor bias can change which landing pad
            # meets a fast walking deck first; final descent therefore belongs
            # solely to the deterministic camera/IMU safety controller.
            residual_envelope = float(
                np.clip(
                    (relative_height - LANDING_POLICY_RESIDUAL_CUTOFF_HEIGHT_M)
                    / (
                        LANDING_POLICY_RESIDUAL_FULL_HEIGHT_M
                        - LANDING_POLICY_RESIDUAL_CUTOFF_HEIGHT_M
                    ),
                    0.0,
                    1.0,
                )
            )
            raw_action_world = drone_rotation[:2, :2] @ action
            if horizontal_error > 1.0e-6:
                inward_direction = relative_world[:2] / horizontal_error
                inward_component = max(
                    0.0,
                    float(raw_action_world @ inward_direction),
                )
                action_world = (
                    self.policy_residual_speed_mps
                    * residual_envelope
                    * inward_component
                    * inward_direction
                )
            else:
                action_world = np.zeros(2, dtype=np.float64)
            desired_xy = self._qr_target_velocity_world[:2] + 0.92 * relative_world[:2] + action_world
            desired_norm = float(np.linalg.norm(desired_xy))
            if desired_norm > 3.60:
                desired_xy *= 3.60 / desired_norm
        else:
            # No QR means no target direction is known.  Follow a predeclared
            # mission search around world origin using own GNSS/EKF position.
            horizontal_error = float("inf")
            relative_height = float("inf")
            desired_xy = self._search_velocity(own_position)
        if (
            self._landing_retry_active
            and detected
            and relative_height >= LANDING_RETRY_REACQUIRE_DEPTH_M
        ):
            # A real camera has reacquired enough depth for another measured
            # alignment pass.  Clear the time/IMU-only retry state; descent
            # must earn a fresh consecutive visual alignment streak.
            self._landing_retry_active = False
            self._landing_committed = False
            self._aligned_streak = 0
        relative_speed = float(np.linalg.norm(self._qr_relative_velocity_world[:2])) if detected else float("inf")
        aligned = detected and horizontal_error < float(self.profile["alignment"]) and relative_speed < 0.45
        # Alignment is intentionally counted at the 100 ms policy rate, not
        # every 5 ms MuJoCo substep.  A descent therefore requires 0.5 s of
        # persistent centering while the Go2 is walking.
        if update_alignment:
            self._aligned_streak = self._aligned_streak + 1 if aligned else 0
            if self._aligned_streak >= 4:
                self._landing_committed = True
            if (
                self._landing_committed
                and detected
                and relative_height <= FINAL_APPROACH_START_HEIGHT_M
                and horizontal_error <= FINAL_APPROACH_ALIGNMENT_M
            ):
                # Enter final approach before the corrected 225.6 mm stock-skid
                # touchdown plane.  Camera depth remains about 161 mm at rest,
                # safely outside the 0.10 m near clip; memory only bridges a
                # genuine detector dropout or brief occlusion.
                self._final_approach = True
                self._final_approach_started = now
            # A moving deck can momentarily catch only the leading gear pair.
            # Lift clear of the high-friction plate, re-center in free flight,
            # and retry instead of remaining snagged at the deck edge.
            if (
                self._landing_committed
                and detected
                and relative_height < FINAL_APPROACH_START_HEIGHT_M + 0.04
                and horizontal_error > 0.045
            ):
                self._approach_recovery = True
            if self._final_approach and not final_memory_valid and not detected:
                self._final_approach = False
                self._landing_committed = False
                self._aligned_streak = 0
        descent = 0.0
        if self._landing_committed and not self._landing_retry_active:
            if relative_height > 1.00:
                descent = 0.85
            elif relative_height > 0.35:
                descent = 0.50
            else:
                descent = 0.22
        # Approach the stock X500's measured 225.6 mm centre-to-QR geometry
        # slowly.  This uses camera PnP only; stock X500 has no gear sensor.
        relative_vertical_velocity = -descent
        if self._landing_committed and relative_height <= FINAL_PRECISION_DESCENT_HEIGHT_M:
            if relative_height > FINAL_PRECISION_TARGET_HEIGHT_M + 0.016:
                relative_vertical_velocity = -0.12
            else:
                relative_vertical_velocity = max(
                    -0.025,
                    4.0 * (FINAL_PRECISION_TARGET_HEIGHT_M - relative_height),
                )
        if final_memory_valid and not detected:
            # Preserve the last visual target velocity through a brief QR
            # dropout.  The corrected stock skid plane leaves the camera well
            # above its near clip at touchdown; this is dropout handling, not
            # a fictitious contact or target-state measurement.
            blind_descent_speed = (
                FINAL_RETRY_BLIND_DESCENT_SPEED_MPS
                if self._landing_retry_count >= 3
                else FINAL_BLIND_DESCENT_SPEED_MPS
            )
            relative_vertical_velocity = -blind_descent_speed
        if self._landing_retry_active:
            # No landing-leg sensor is involved.  Following an IMU impact that
            # did not settle, keep the last camera-derived horizontal motion
            # and climb until the QR is optically measurable again.
            relative_vertical_velocity = LANDING_RETRY_CLIMB_SPEED_MPS
        if self._approach_recovery:
            relative_vertical_velocity = (
                0.18 if relative_height < FINAL_PRECISION_DESCENT_HEIGHT_M else 0.0
            )
            if horizontal_error < 0.025 and relative_height > FINAL_PRECISION_TARGET_HEIGHT_M + 0.045:
                self._approach_recovery = False
                relative_vertical_velocity = -0.12
        if tracking:
            desired_vertical_velocity = self._qr_target_velocity_world[2] + relative_vertical_velocity
        else:
            # Search height is commanded from own GNSS/barometric estimate.
            desired_vertical_velocity = float(np.clip(1.8 * (self._search_altitude - own_position[2]), -0.80, 1.20))
        desired_velocity = np.array(
            [desired_xy[0], desired_xy[1], desired_vertical_velocity], dtype=np.float64
        )
        velocity_error = desired_velocity - own_velocity
        acceleration = np.array([4.8 * velocity_error[0], 4.8 * velocity_error[1], 16.0 * velocity_error[2]], dtype=np.float64)
        force = self.drone_mass * (acceleration - self.model.opt.gravity)
        self.data.xfrc_applied[self.drone_id] = 0.0
        self.data.xfrc_applied[self.drone_id, :3] = np.clip(force, -34.0, 34.0)
        # Above the approach gate the X500 stays world-level.  Near the deck
        # it matches the complete QR-mount orientation (including yaw), so both
        # continuous skid rails meet the walking, tilted surface on one plane.
        target_rotation = (
            self._qr_target_rotation_world
            if tracking and relative_height < 0.45
            else np.eye(3, dtype=np.float64)
        )
        attitude_error = 0.5 * sum(
            (np.cross(drone_rotation[:, axis], target_rotation[:, axis]) for axis in range(3)),
            start=np.zeros(3, dtype=np.float64),
        )
        angular_velocity = self._onboard_angular_velocity()
        attitude_torque = 12.0 * attitude_error - 1.40 * angular_velocity
        if self._imu_impact_latched:
            # Keep the camera-derived horizontal velocity loop active while
            # softening only the vertical channel.  Zeroing XY thrust here
            # would make a landed aircraft brake in the world frame while
            # the walking deck continues at up to 1.1 m/s, loading both the
            # X500 legs and Go2 with a fictitious drag impulse.
            # Cut collective slightly below hover for the 0.35 s IMU settle
            # window.  Increasing thrust to arrest the pre-impact descent
            # would add to the normal impulse and bounce the leading skid
            # pair.  This uses only the latched onboard IMU state and the
            # vehicle's known mass/gravity calibration, never MuJoCo contact.
            self.data.xfrc_applied[self.drone_id, 2] = (
                IMU_SETTLE_THRUST_FRACTION
                * self.drone_mass
                * abs(float(self.model.opt.gravity[2]))
            )
            # Keep the last solvePnP deck attitude as a compliant reference so
            # a leading rail edge can settle into bilateral skid support.
            attitude_torque = 8.0 * attitude_error - 1.80 * angular_velocity
        # PX4 knows its own commanded collective thrust.  Cache the predicted
        # body-Z specific force before the physics step so the IMU landing
        # detector can reject ordinary commanded acceleration.
        self._commanded_specific_force_body_z = float(
            (drone_rotation.T @ self.data.xfrc_applied[self.drone_id, :3])[2]
            / self.drone_mass
        )
        self.data.xfrc_applied[self.drone_id, 3:6] = np.clip(attitude_torque, -8.0, 8.0)

    def _update_imu_landing_state(self) -> None:
        """Latch physical impact and retry using stock IMU/EKF channels only.

        This is the PX4-style alternative to a nonexistent landing-leg
        sensor.  Stable simulator contact terminates the episode before the
        short settle timer expires.  If it does not settle, the same onboard
        state machine climbs to reacquire QR rather than assuming touchdown.
        """
        now = float(self.data.time)
        recent_final_vision = (
            self._final_approach
            and now - self._last_qr_seen_time <= FINAL_APPROACH_MEMORY_S
        )
        if not self._imu_impact_latched and not self._landing_retry_active and recent_final_vision:
            if self._qr_detected:
                relative_world = self._onboard_rotation() @ self._qr_translation_body
            else:
                propagated_target = (
                    self._qr_target_position_world
                    + max(0.0, now - self._last_qr_seen_time) * self._qr_target_velocity_world
                )
                relative_world = propagated_target - self._onboard_position()
            visual_relative_height = max(0.0, -float(relative_world[2]))
            # The deck-normal touchdown impulse appears on the body-Z IMU
            # axis.  Subtract the body-Z specific force predicted by the
            # controller's own thrust command; otherwise an ordinary climb
            # arrest could be misclassified as a physical impact.
            vertical_specific_force = float(self._sensor("drone_accelerometer")[2])
            impact_innovation = (
                vertical_specific_force - self._commanded_specific_force_body_z
            )
            vertical_speed = abs(float(self._onboard_velocity()[2]))
            if (
                visual_relative_height <= IMU_IMPACT_MAX_VISUAL_HEIGHT_M
                and impact_innovation >= IMU_IMPACT_DELTA_MPS2
                and vertical_speed <= 0.45
            ):
                self._imu_impact_latched = True
                self._imu_impact_time = now
        if self._imu_impact_latched and now - self._imu_impact_time >= IMU_SETTLE_TIME_S:
            self._imu_impact_latched = False
            self._landing_retry_active = True
            self._landing_retry_count += 1
            self._final_approach = False
            self._landing_committed = False
            self._approach_recovery = False
            self._aligned_streak = 0

    def _apply_wind_disturbance(self) -> None:
        """Apply seeded wind as a physical wrench, not a sensor/control lie."""
        persistence = 0.992
        sigma_force = self.drone_mass * float(self.profile["wind"])
        innovation = self.np_random.normal(
            0.0, sigma_force * math.sqrt(1.0 - persistence * persistence), size=2
        )
        self._wind_force_world[:2] = persistence * self._wind_force_world[:2] + innovation
        self.data.xfrc_applied[self.drone_id, :2] += self._wind_force_world[:2]

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, float]]:
        super().reset(seed=seed)
        if self.terrain_task == "rough":
            self._active_rough_level = (
                self._requested_rough_level
                if self._requested_rough_level is not None
                else int(self.np_random.integers(1, 4))
            )
            configure_rough_terrain(self.model, level=self._active_rough_level)
        mujoco.mj_resetData(self.model, self.data)
        initial_ground_height = terrain_height_at(
            self.terrain_task, 0.0, 0.0, rough_level=self._active_rough_level if self.terrain_task == "rough" else None
        )
        half_pitch = 0.5 * terrain_initial_pitch_rad(self.terrain_task)
        self.data.qpos[:7] = (
            0.0, 0.0, initial_ground_height + 0.32,
            math.cos(half_pitch), 0.0, math.sin(half_pitch), 0.0,
        )
        self.data.qpos[self.go2_qposadr] = GO2_STAND_POSE
        radius_low, radius_high = self.profile["radius"]  # type: ignore[misc]
        altitude_low, altitude_high = self.profile["altitude"]  # type: ignore[misc]
        radius = float(self.np_random.uniform(radius_low, radius_high))
        heading = float(self.np_random.uniform(-math.pi, math.pi))
        altitude = float(self.np_random.uniform(altitude_low, altitude_high))
        # A forward path starts immediately, so initialise around the QR site
        # after the Go2 home pose has been forwarded once.
        mujoco.mj_forward(self.model, self.data)
        pad = self.pad_position
        self.data.qpos[self.drone_qposadr:self.drone_qposadr + 7] = (pad[0] + radius * math.cos(heading), pad[1] + radius * math.sin(heading), pad[2] + altitude, 1.0, 0.0, 0.0, 0.0)
        self.data.qvel[:] = 0.0
        self._step_count = 0
        self._aligned_streak = 0
        self._landing_committed = False
        self._approach_recovery = False
        self._dropout = False
        self._offline_sim_landing_skid_contact_count = 0
        self._offline_sim_landing_normal_force = 0.0
        self._offline_sim_max_contact_penetration = 0.0
        self._touchdown_success_evidence = False
        self._imu_impact_latched = False
        self._imu_impact_time = float("-inf")
        self._commanded_specific_force_body_z = abs(float(self.model.opt.gravity[2]))
        self._landing_retry_active = False
        self._landing_retry_count = 0
        self._path_length = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._next_estimator_time = 0.0
        self._next_camera_time = 0.0
        self._update_onboard_estimator(force=True)
        self._last_qr_seen_time = float("-inf")
        self._final_approach = False
        self._final_approach_started = float("-inf")
        self._search_origin_xy[:] = SEARCH_AREA_CENTER_WORLD
        # The search waypoint is declared before take-off from the mission's
        # known terrain map at x=0, not read from the moving QR/Go2 state.
        # This keeps the X500 at a usable camera-search height when the
        # downhill 10% course begins several metres above world z=0.
        self._search_altitude = initial_ground_height + SEARCH_ALTITUDE_WORLD_M
        self._search_started = float(self.data.time)
        self._wind_force_world[:] = 0.0
        self._prev_pad_position = self.pad_position
        self._previous_base_position = self.base_position
        self._previous_go2_foot_positions = self.data.geom_xpos[self.go2_foot_geom_ids].copy()
        self._previous_go2_contact_mask = self._go2_ground_contact_mask()
        self._go2_stance_slip_mps = 0.0
        self._pad_velocity[:] = self._path_command(0.0)[:2]
        self._pad_vertical_velocity = 0.0
        self._qr_detected = False
        self._qr_center_norm[:] = 0.0
        self._qr_translation_body[:] = 0.0
        self._qr_rotation_body[:] = np.eye(3)
        self._qr_center_rate[:] = 0.0
        self._qr_relative_velocity_world[:] = 0.0
        self._qr_target_velocity_world[:] = 0.0
        self._qr_target_position_world[:] = 0.0
        self._qr_target_rotation_world[:] = np.eye(3)
        self._previous_qr_center[:] = 0.0
        self._previous_qr_relative_world[:] = 0.0
        self._previous_qr_measurement_time = float(self.data.time)
        self._previous_qr_valid = False
        self._qr_depth = 0.0
        self._update_qr_camera_measurement(force=True)
        if self._legged_loco is not None:
            self._legged_loco.reset(self)
        return self._observation(), {
            "radius_m": radius,
            "altitude_m": self._relative_altitude(),
            "go2_pad_mass_kg": 0.22,
            "terrain_task": self.terrain_task,
            "rough_level": float(self._active_rough_level) if self.terrain_task == "rough" else 0.0,
        }

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, float]]:
        self._step_count += 1
        self._touchdown_success_evidence = False
        previous_distance = float(np.linalg.norm(self._horizontal_error()))
        for physics_index in range(self.physics_steps):
            self._apply_go2_locomotion()
            self._drone_control(action, update_alignment=physics_index == 0)
            self._apply_wind_disturbance()
            mujoco.mj_step(self.model, self.data)
            self._update_onboard_estimator()
            self._update_imu_landing_state()
            current_pad = self.pad_position
            instantaneous_pad_velocity = (current_pad - self._prev_pad_position) / float(self.model.opt.timestep)
            # The X500 must follow translational deck motion, not every 5 ms
            # footfall vibration of the dorsal site.  A 40 ms low-pass keeps
            # the feed-forward physically meaningful and prevents edge snags.
            self._pad_velocity[:] = 0.88 * self._pad_velocity + 0.12 * instantaneous_pad_velocity[:2]
            self._pad_vertical_velocity = 0.88 * self._pad_vertical_velocity + 0.12 * float(instantaneous_pad_velocity[2])
            self._prev_pad_position = current_pad
            current_base = self.base_position
            self._path_length += float(np.linalg.norm(current_base[:2] - self._previous_base_position[:2]))
            self._previous_base_position = current_base
            count, normal, penetration = self._contact_calibration()
            self._offline_sim_landing_skid_contact_count = count
            self._offline_sim_landing_normal_force = normal
            self._offline_sim_max_contact_penetration = max(
                self._offline_sim_max_contact_penetration, penetration
            )
            foot_positions = self.data.geom_xpos[self.go2_foot_geom_ids].copy()
            foot_velocity = (foot_positions - self._previous_go2_foot_positions) / float(self.model.opt.timestep)
            go2_contacts = self._go2_ground_contact_mask()
            stable_go2_contacts = go2_contacts & self._previous_go2_contact_mask
            if np.any(stable_go2_contacts):
                instantaneous_slip = float(np.mean(np.linalg.norm(foot_velocity[stable_go2_contacts, :2], axis=1)))
                self._go2_stance_slip_mps = 0.85 * self._go2_stance_slip_mps + 0.15 * instantaneous_slip
            else:
                self._go2_stance_slip_mps *= 0.98
            self._previous_go2_foot_positions = foot_positions
            self._previous_go2_contact_mask = go2_contacts
            self._update_qr_camera_measurement()
            if self.physics_observer is not None:
                self.physics_observer(self)
            # Contact is simulator-only termination/evaluation evidence, not
            # an X500 sensor or a controller input.  Check it at the native
            # 5 ms physics rate so a valid moving-deck touchdown is not lost
            # merely because the 100 ms policy boundary lands on a bounce.
            instantaneous_relative_velocity = np.array(
                [
                    self.drone_velocity[0] - self._pad_velocity[0],
                    self.drone_velocity[1] - self._pad_velocity[1],
                    self.drone_velocity[2] - self._pad_vertical_velocity,
                ],
                dtype=np.float64,
            )
            if (
                count >= 2
                and float(np.linalg.norm(instantaneous_relative_velocity)) < 0.40
                and self._relative_altitude() <= SUCCESS_MAX_RELATIVE_HEIGHT_M
                and float(np.linalg.norm(self._horizontal_error())) < float(self.profile["landing"])
            ):
                self._touchdown_success_evidence = True
                break
        self._dropout = bool(self.np_random.random() < float(self.profile["dropout"]))
        horizontal_distance = float(np.linalg.norm(self._horizontal_error()))
        relative_altitude = self._relative_altitude()
        # Simulator-only touchdown scoring must use velocity relative to the
        # moving deck.  World speed is intentionally high (0.7--1.1 m/s), so
        # treating it as landing impact speed would reject a perfectly settled
        # vehicle that is simply being carried by Go2.
        relative_landing_velocity = np.array(
            [
                self.drone_velocity[0] - self._pad_velocity[0],
                self.drone_velocity[1] - self._pad_velocity[1],
                self.drone_velocity[2] - self._pad_vertical_velocity,
            ],
            dtype=np.float64,
        )
        relative_landing_speed = float(np.linalg.norm(relative_landing_velocity))
        base_up = float(self.data.xmat[self.base_id, 8])
        go2_tilt_deg = math.degrees(math.acos(float(np.clip(base_up, -1.0, 1.0))))
        # Physical contacts are simulator-only scoring evidence.  They are
        # never exposed to the stock X500 policy or flight controller.
        stable_contact = (
            self._offline_sim_landing_skid_contact_count >= 2
            and relative_landing_speed < 0.40
        )
        success = self._touchdown_success_evidence or (
            stable_contact
            and relative_altitude <= SUCCESS_MAX_RELATIVE_HEIGHT_M
            and horizontal_distance < float(self.profile["landing"])
        )
        hard_landing = (
            relative_altitude <= DEEP_PENETRATION_HEIGHT_M
            or (
                relative_altitude <= OFF_CENTRE_HARD_LANDING_HEIGHT_M
                and horizontal_distance >= float(self.profile["landing"])
                and not self._imu_impact_latched
                and not self._landing_retry_active
            )
        )
        go2_fall = self.base_position[2] < 0.18 or base_up < 0.55
        out_of_bounds = horizontal_distance > 15.0 or self.drone_position[2] > 9.0
        terminated = success or hard_landing or go2_fall or out_of_bounds
        truncated = self._step_count >= self.max_steps
        # The deterministic camera servo already supplies the nominal
        # correction.  Penalize unnecessary raw residual proposals strongly
        # so off-policy actors learn to stay near zero instead of exploiting
        # tiny timing changes in the moving-deck contact phase.
        reward = 7.0 * (previous_distance - horizontal_distance) - 0.035 * horizontal_distance - 1.0 * float(np.square(action).sum())
        if horizontal_distance < 0.30:
            reward += 0.12
        # Do not shape the dense reward with a fictitious landing-leg sensor.
        # MuJoCo contact is used only to label a completed physical landing,
        # terminate the simulated episode, and populate offline diagnostics.
        if success:
            reward += 110.0
        elif hard_landing:
            reward -= 55.0
        elif go2_fall:
            reward -= 80.0
        elif out_of_bounds:
            reward -= 25.0
        info = {
            "horizontal_error_m": horizontal_distance,
            "altitude_m": relative_altitude,
            "success": float(success),
            "hard_landing": float(hard_landing),
            "go2_fall": float(go2_fall),
            "aligned_streak": float(self._aligned_streak),
            "landing_committed": float(self._landing_committed),
            "approach_recovery": float(self._approach_recovery),
            "imu_impact_latched": float(self._imu_impact_latched),
            "landing_retry_active": float(self._landing_retry_active),
            "landing_retry_count": float(self._landing_retry_count),
            "qr_detected": float(self._qr_detected),
            "qr_pnp_depth_m": self._qr_depth if self._qr_detected else 0.0,
            "pad_speed_mps": float(np.linalg.norm(self._pad_velocity)),
            "go2_path_distance_m": self._path_length,
            "go2_speed_mps": float(np.linalg.norm(self.data.qvel[:2])),
            "offline_sim_relative_landing_speed_mps": relative_landing_speed,
            "go2_stance_slip_mps": self._go2_stance_slip_mps,
            "go2_base_height_m": float(self.base_position[2]),
            "go2_tilt_deg": go2_tilt_deg,
            "terrain_ground_height_m": terrain_height_at(
                self.terrain_task,
                float(self.base_position[0]),
                float(self.base_position[1]),
                rough_level=self._active_rough_level if self.terrain_task == "rough" else None,
            ),
            "terrain_rough_level": float(self._active_rough_level) if self.terrain_task == "rough" else 0.0,
            "go2_assist_force_n": float(np.linalg.norm(self.data.xfrc_applied[self.base_id, :2])),
            "offline_sim_landing_skid_contacts": float(
                self._offline_sim_landing_skid_contact_count
            ),
            "offline_sim_landing_normal_force_n": self._offline_sim_landing_normal_force,
            "offline_sim_max_contact_penetration_m": self._offline_sim_max_contact_penetration,
            "episode_steps": float(self._step_count),
            "physics_backend": "mujoco_official_unitree_go2",
            "locomotion_backend": "legged-loco-mujoco-ppo" if self._legged_loco is not None else "unitree-reference-trot",
        }
        return self._observation(), float(reward), terminated, truncated, info

    def close(self) -> None:
        return None
