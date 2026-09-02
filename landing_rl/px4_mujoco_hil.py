"""External PX4 SITL ↔ MuJoCo HIL bridge for the flat X500 demonstration.

This module deliberately runs the already-built PX4 SITL executable as a
separate process.  MuJoCo sends HIL IMU/barometer/GPS measurements through
PX4's ``simulator_mavlink`` TCP interface; PX4 EKF2 then produces the state
used by the companion controller.  The bridge returns PX4's four
``HIL_ACTUATOR_CONTROLS`` motor values and converts *only those values* into
the physical X500 wrench applied to MuJoCo.

It does not edit or build the PX4 source tree.  Every SITL run receives a
temporary rootfs, so its parameters, logs and data manager state cannot alter
an existing PX4/Gazebo run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import TextIO

import numpy as np
from pymavlink import mavutil


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PX4_BUILD = PROJECT_ROOT / "PX4-Autopilot" / "build" / "px4_sitl_default"
PX4_BINARY = PX4_BUILD / "bin" / "px4"

# This MuJoCo scene uses a body-aligned NWU world: X is north/forward, Y is
# left (therefore west while facing north), Z is up.  The stock X500 body is
# FLU; PX4 HIL messages are NED/FRD.  Treating MuJoCo +Y as east reverses GPS
# longitude, local-position estimates and lateral Offboard motion.
WORLD_NWU_TO_NED = np.diag((1.0, -1.0, -1.0))
FLU_TO_FRD = np.diag((1.0, -1.0, -1.0))

# The output order is the PX4 X500 control-allocation order configured by
# airframe 4001.  Rotor 0 is front-right in PX4 FRD, which becomes
# front-right (negative Y) in the MuJoCo FLU model; the rest follow 0: FR,
# 1: RL, 2: FL, 3: RR.  Keeping this mapping exact is essential: a
# permutation can leave collective thrust plausible while reversing PX4's
# roll/pitch torque.
X500_ROTOR_POSITIONS_FLU_M = np.array(
    (
        (0.174, -0.174, 0.060),
        (-0.174, 0.174, 0.060),
        (0.174, 0.174, 0.060),
        (-0.174, -0.174, 0.060),
    ),
    dtype=np.float64,
)
# Airframe 4001 uses +,+,-,- yaw moment ratios in PX4's FRD axes.  Convert
# those torques to the MuJoCo FLU body-Z convention by negating the sign.
X500_YAW_MOMENT_RATIO_FLU = np.array((-0.05, -0.05, 0.05, 0.05), dtype=np.float64)
# Airframe 4001's PX4 multicopter controller uses the SITL default
# MPC_THR_HOVER=0.50.  Calibrate the MuJoCo force conversion at that actual
# PX4 actuator fraction so a level 0.50 command balances the measured X500
# mass instead of introducing a fictitious 17% descent bias.
X500_HOVER_THRUST_FRACTION = 0.50
HIL_GPS_ORIGIN_ALTITUDE_MSL_M = 488.0

# MAVLink HIL_SENSOR publishes accel/gyro/mag plus absolute pressure,
# differential pressure, pressure altitude and temperature through bits 0–12.
# PX4's simulator_mavlink accepts a barometer only when the complete BARO
# mask (pressure, altitude and temperature) is set.
HIL_SENSOR_ALL_FIELDS = (1 << 13) - 1
PX4_CUSTOM_MAIN_MODE_OFFBOARD = 6
# SET_POSITION_TARGET_LOCAL_NED: ignore position (bits 0–2), acceleration
# (6–8), yaw and yaw-rate (10–11), while deliberately *using* velocity
# components VX/VY/VZ (bits 3–5 must stay clear).
VELOCITY_ONLY_MASK = 0b0000110111000111


@dataclass
class Px4EkfState:
    """Latest EKF2 MAVLink state, in PX4 local NED coordinates."""

    local_position_ned_m: np.ndarray | None = None
    local_velocity_ned_mps: np.ndarray | None = None
    attitude_rpy_ned_rad: np.ndarray | None = None
    heartbeat_seen: bool = False
    custom_mode: int = 0
    local_position_messages: int = 0
    attitude_messages: int = 0
    horizontal_innovation_ratio: float | None = None
    vertical_innovation_ratio: float | None = None
    velocity_innovation_ratio: float | None = None
    solution_status_flags: int = 0
    estimator_status_messages: int = 0


@dataclass
class Px4HilDiagnostics:
    """Audit counters for the external controller path."""

    hil_sensor_messages: int = 0
    hil_gps_messages: int = 0
    offboard_velocity_messages: int = 0
    offboard_send_failures: int = 0
    last_offboard_target: tuple[int, int] | None = None
    last_hil_sensor_time_usec: int = 0
    last_hil_gps_time_usec: int = 0
    hil_time_regressions: int = 0
    actuator_messages: int = 0
    gcs_messages: int = 0
    actuator_armed_messages: int = 0
    last_command_ack: tuple[int, int] | None = None
    command_ack_results: dict[int, int] = field(default_factory=dict)
    last_status_text: str = ""
    status_texts: list[str] = field(default_factory=list)
    sys_status_present: int = 0
    sys_status_enabled: int = 0
    sys_status_health: int = 0
    last_actuator_controls: np.ndarray = field(
        default_factory=lambda: np.zeros(4, dtype=np.float64)
    )


class Px4MujocoHilSession:
    """Manage one isolated PX4 SITL instance and its MAVLink HIL links."""

    def __init__(
        self,
        *,
        simulator_tcp_port: int = 4560,
        offboard_udp_port: int = 14540,
        log_path: Path | None = None,
        px4_binary: Path = PX4_BINARY,
    ) -> None:
        self.simulator_tcp_port = int(simulator_tcp_port)
        self.offboard_udp_port = int(offboard_udp_port)
        self.log_path = log_path
        self.px4_binary = Path(px4_binary)
        self._temporary_rootfs: tempfile.TemporaryDirectory[str] | None = None
        self._px4_log: TextIO | None = None
        self.process: subprocess.Popen[bytes] | None = None
        self.hil_link: mavutil.mavfile | None = None
        self.gcs_link: mavutil.mavfile | None = None
        self.target_system: int | None = None
        self.target_component: int | None = None
        self.ekf = Px4EkfState()
        self.diagnostics = Px4HilDiagnostics()
        self._last_gcs_heartbeat_s = float("-inf")
        self._next_gps_time_us = 0
        self._last_hil_sensor_time_us = -1
        self._estimator_status_requested = False
        self.ulog_path: Path | None = None

    @property
    def hil_connected(self) -> bool:
        return self.hil_link is not None and getattr(self.hil_link, "port", None) is not None

    @property
    def armed(self) -> bool:
        return self.diagnostics.actuator_armed_messages > 0

    @property
    def ekf_ready(self) -> bool:
        return (
            self.ekf.local_position_ned_m is not None
            and self.ekf.local_velocity_ned_mps is not None
            and self.ekf.attitude_rpy_ned_rad is not None
        )

    @property
    def offboard_active(self) -> bool:
        """Whether PX4 reports its own Offboard main mode in HEARTBEAT."""
        return ((self.ekf.custom_mode >> 16) & 0xFF) == PX4_CUSTOM_MAIN_MODE_OFFBOARD

    def __enter__(self) -> "Px4MujocoHilSession":
        self.start()
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()

    def start(self) -> None:
        if self.process is not None:
            raise RuntimeError("PX4 HIL session is already running")
        if not self.px4_binary.is_file():
            raise FileNotFoundError(
                "PX4 SITL binary is missing. Build PX4 first or pass px4_binary: "
                f"{self.px4_binary}"
            )
        if not (PX4_BUILD / "etc" / "init.d-posix" / "rcS").is_file():
            raise FileNotFoundError(f"PX4 SITL ROMFS is missing under {PX4_BUILD}")

        self._temporary_rootfs = tempfile.TemporaryDirectory(prefix="px4_mujoco_hil_")
        rootfs = Path(self._temporary_rootfs.name)
        # Airframe 4001 normally starts Gazebo-gz.  Copy only the generated
        # runtime ROMFS into this ephemeral rootfs and patch that *copy* to
        # select PX4's built-in simulator_mavlink HIL backend instead.  The
        # checked-out PX4 tree (which can contain a user's Gazebo work) stays
        # untouched.
        shutil.copytree(PX4_BUILD / "etc", rootfs / "etc", symlinks=True)
        x500_airframe = rootfs / "etc" / "init.d-posix" / "airframes" / "4001_gz_x500"
        x500_text = x500_airframe.read_text(encoding="utf-8")
        x500_text = x500_text.replace(
            "PX4_SIMULATOR=${PX4_SIMULATOR:=gz}",
            "# MuJoCo HIL selects simulator_mavlink; never start Gazebo here.",
        ).replace(
            "param set-default SIM_GZ_EN 1",
            "param set-default SIM_GZ_EN 0",
        ).replace(
            "param set-default NAV_DLL_ACT 2",
            "param set-default NAV_DLL_ACT 0",
        )
        # simulator_mavlink subscribes to pwm_out_sim, whose SITL parameter
        # prefix is PWM_MAIN.  The Gazebo X500 config only declares
        # SIM_GZ_EC functions, so declare the same four Motor functions for
        # the HIL-only copy as well.
        x500_text += """
param set-default PWM_MAIN_FUNC1 101
param set-default PWM_MAIN_FUNC2 102
param set-default PWM_MAIN_FUNC3 103
param set-default PWM_MAIN_FUNC4 104
# Keep PX4's full GNSS quality checks, but reduce the mandatory continuous
# health observation window from the hardware-oriented 10 s to 1 s for this
# deterministic, fixed-seed HIL scene.  This only shortens warm-up; it does
# not suppress a failed fix/accuracy/drift check.
param set-default EKF2_REQ_GPS_H 1.0
# The landing comparison deliberately starts the X500 in free flight, rather
# than supported on a launch pad.  Skip PX4's ground-launch thrust ramp in
# this temporary HIL-only configuration so the normal multicopter position
# controller immediately supplies hover collective after arming.
param set-default MPC_TKO_RAMP_T 0.0
"""
        x500_airframe.write_text(x500_text, encoding="utf-8")
        (rootfs / "eeprom").mkdir()
        (rootfs / "log").mkdir()

        # Start both listeners *before* PX4.  simulator_mavlink connects to
        # the TCP listener, while the normal PX4 API instance sends the GCS
        # heartbeat and LOCAL_POSITION_NED stream to the UDP listener.
        self.hil_link = mavutil.mavlink_connection(
            f"tcpin:127.0.0.1:{self.simulator_tcp_port}",
            source_system=1,
            source_component=1,
        )
        self.gcs_link = mavutil.mavlink_connection(
            f"udpin:127.0.0.1:{self.offboard_udp_port}",
            source_system=245,
            source_component=190,
        )

        environment = os.environ.copy()
        environment.update(
            {
                # Use the stock PX4 X500 allocation but deliberately choose
                # simulator_mavlink instead of the Gazebo-gz backend.
                "PX4_SIM_MODEL": "none",
                "PX4_SYS_AUTOSTART": "4001",
                "PX4_SIMULATOR": "mavlink",
                "PX4_SIM_HOSTNAME": "127.0.0.1",
                "PX4_PARAM_SIM_GZ_EN": "0",
                # The simulated HIL GPS is present, but this guard makes an
                # EKF warm-up delay fail safe rather than forcing the runner
                # to bypass PX4 arming checks with a custom wrench.
                "PX4_PARAM_COM_ARM_WO_GPS": "1",
            }
        )
        if self.log_path is None:
            self._px4_log = open(os.devnull, "w", encoding="utf-8")
        else:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._px4_log = self.log_path.open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            [
                str(self.px4_binary),
                "-d",
                str(rootfs),
                "-s",
                "etc/init.d-posix/rcS",
                "-i",
                "0",
            ],
            cwd=rootfs,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=self._px4_log,
            stderr=subprocess.STDOUT,
        )

    def close(self) -> None:
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5.0)
            self.process = None
        if self.hil_link is not None:
            self.hil_link.close()
            self.hil_link = None
        if self.gcs_link is not None:
            self.gcs_link.close()
            self.gcs_link = None
        if self._px4_log is not None:
            self._px4_log.close()
            self._px4_log = None
        if self._temporary_rootfs is not None:
            # Preserve the actual PX4 ULog beside the optional text log
            # before removing this run's isolated rootfs.  A dashboard can
            # therefore audit EKF2 and actuator output without retaining a
            # mutable PX4 work directory.
            if self.log_path is not None:
                run_root = Path(self._temporary_rootfs.name)
                ulgs = sorted(run_root.glob("log/**/*.ulg"))
                if ulgs:
                    self.ulog_path = self.log_path.with_suffix(".ulg")
                    shutil.copy2(ulgs[-1], self.ulog_path)
            self._temporary_rootfs.cleanup()
            self._temporary_rootfs = None

    def assert_running(self) -> None:
        if self.process is None:
            raise RuntimeError("PX4 HIL session has not been started")
        return_code = self.process.poll()
        if return_code is not None:
            raise RuntimeError(f"PX4 SITL exited unexpectedly with status {return_code}")

    def pump(self) -> None:
        """Drain PX4 motor commands and EKF telemetry without blocking."""
        self.assert_running()
        if self.hil_link is not None:
            while True:
                message = self.hil_link.recv_match(blocking=False)
                if message is None:
                    break
                if message.get_type() == "HIL_ACTUATOR_CONTROLS":
                    controls = np.asarray(message.controls[:4], dtype=np.float64)
                    self.diagnostics.last_actuator_controls[:] = self._normalise_controls(controls)
                    self.diagnostics.actuator_messages += 1
                    if int(message.mode) & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
                        self.diagnostics.actuator_armed_messages += 1
        if self.gcs_link is not None:
            while True:
                message = self.gcs_link.recv_match(blocking=False)
                if message is None:
                    break
                self.diagnostics.gcs_messages += 1
                message_type = message.get_type()
                if message_type == "HEARTBEAT":
                    # ``udpin`` can also receive a GCS heartbeat echo.  It
                    # must never replace the autopilot destination with the
                    # sender's GCS component; doing so silently turns the
                    # next offboard setpoint into a packet addressed to us.
                    if int(getattr(message, "autopilot", mavutil.mavlink.MAV_AUTOPILOT_INVALID)) != mavutil.mavlink.MAV_AUTOPILOT_INVALID:
                        self.target_system = int(message.get_srcSystem())
                        self.target_component = int(message.get_srcComponent())
                        self.ekf.heartbeat_seen = True
                        self.ekf.custom_mode = int(getattr(message, "custom_mode", 0))
                elif message_type == "LOCAL_POSITION_NED":
                    self.ekf.local_position_ned_m = np.array(
                        (float(message.x), float(message.y), float(message.z)), dtype=np.float64
                    )
                    self.ekf.local_velocity_ned_mps = np.array(
                        (float(message.vx), float(message.vy), float(message.vz)), dtype=np.float64
                    )
                    self.ekf.local_position_messages += 1
                elif message_type == "ATTITUDE":
                    self.ekf.attitude_rpy_ned_rad = np.array(
                        (float(message.roll), float(message.pitch), float(message.yaw)), dtype=np.float64
                    )
                    self.ekf.attitude_messages += 1
                elif message_type == "ESTIMATOR_STATUS":
                    self.ekf.velocity_innovation_ratio = float(message.vel_ratio)
                    self.ekf.horizontal_innovation_ratio = float(message.pos_horiz_ratio)
                    self.ekf.vertical_innovation_ratio = float(message.pos_vert_ratio)
                    self.ekf.solution_status_flags = int(message.flags)
                    self.ekf.estimator_status_messages += 1
                elif message_type == "COMMAND_ACK":
                    self.diagnostics.last_command_ack = (
                        int(message.command), int(message.result)
                    )
                    self.diagnostics.command_ack_results[int(message.command)] = int(message.result)
                elif message_type == "STATUSTEXT":
                    text = getattr(message, "text", "")
                    self.diagnostics.last_status_text = (
                        text.decode(errors="replace") if isinstance(text, bytes) else str(text)
                    ).rstrip("\x00")
                    self.diagnostics.status_texts.append(self.diagnostics.last_status_text)
                    del self.diagnostics.status_texts[:-32]
                elif message_type == "SYS_STATUS":
                    self.diagnostics.sys_status_present = int(message.onboard_control_sensors_present)
                    self.diagnostics.sys_status_enabled = int(message.onboard_control_sensors_enabled)
                    self.diagnostics.sys_status_health = int(message.onboard_control_sensors_health)
        if (
            not self._estimator_status_requested
            and self.gcs_link is not None
            and self.target_system is not None
            and self.target_component is not None
        ):
            # COMMAND_LONG/MAV_CMD_SET_MESSAGE_INTERVAL asks the normal PX4
            # MAVLink module for its own EKF2 innovation report.  This is
            # diagnostic telemetry only; it never alters the estimator or
            # actuator path.
            self.gcs_link.mav.command_long_send(
                self.target_system,
                self.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                0,
                mavutil.mavlink.MAVLINK_MSG_ID_ESTIMATOR_STATUS,
                100_000.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            )
            self._estimator_status_requested = True

    @staticmethod
    def _normalise_controls(controls: np.ndarray) -> np.ndarray:
        """Handle PX4's normal [0, 1] output and defensive PWM conversion."""
        values = np.asarray(controls, dtype=np.float64)
        if np.nanmax(np.abs(values), initial=0.0) > 1.5:
            values = (values - 1000.0) / 1000.0
        return np.clip(np.nan_to_num(values, nan=0.0), 0.0, 1.0)

    def send_hil_sample(
        self,
        *,
        time_usec: int,
        position_enu_m: np.ndarray,
        velocity_enu_mps: np.ndarray,
        acceleration_body_flu_mps2: np.ndarray,
        angular_velocity_body_flu_radps: np.ndarray,
    ) -> bool:
        """Send one genuine MuJoCo IMU/barometer/GPS measurement batch to PX4."""
        self.pump()
        if self.hil_link is None or not self.hil_connected:
            return False
        position = np.asarray(position_enu_m, dtype=np.float64)
        velocity = np.asarray(velocity_enu_mps, dtype=np.float64)
        acceleration_frd = FLU_TO_FRD @ np.asarray(acceleration_body_flu_mps2, dtype=np.float64)
        gyro_frd = FLU_TO_FRD @ np.asarray(angular_velocity_body_flu_radps, dtype=np.float64)
        # The MuJoCo scene is local ENU.  HIL_GPS uses MSL altitude, so the
        # barometer must use that *same* global reference.  Feeding a local
        # 1.4 m pressure altitude beside a 489.4 m GPS altitude makes a real
        # EKF correctly reject the inconsistent height sources at pre-arm.
        altitude_msl_m = HIL_GPS_ORIGIN_ALTITUDE_MSL_M + float(position[2])
        # ISA barometer model. MAVLink HIL_SENSOR pressure uses hPa.  The
        # real sensor is never bit-for-bit constant; a tiny deterministic
        # measurement variation prevents PX4's sensor voter from correctly
        # classifying a parked HIL barometer as a frozen device during the
        # pre-arm warm-up.  It is < 2 cm altitude-equivalent and does not
        # encode the target or any controller command.
        pressure_hpa = (
            1013.25 * max(0.01, 1.0 - 2.25577e-5 * altitude_msl_m) ** 5.25588
            + 0.0015 * math.sin(float(time_usec) * 7.3e-6)
        )
        temperature_c = 20.0 + 0.002 * math.sin(float(time_usec) * 5.1e-6)
        # Likewise preserve a physically small (sub-milligauss) variation in
        # the HIL magnetometer.  PX4's voter properly rejects a perfectly
        # identical stream as a frozen compass, which would prevent yaw
        # alignment and, in turn, GNSS horizontal fusion.
        magnetic_field_frd_gauss = (
            0.21523 + 0.00008 * math.sin(float(time_usec) * 6.7e-6),
            0.00008 * math.cos(float(time_usec) * 4.9e-6),
            0.42741 + 0.00008 * math.sin(float(time_usec) * 5.9e-6),
        )
        time_usec = int(time_usec)
        if time_usec <= self._last_hil_sensor_time_us:
            self.diagnostics.hil_time_regressions += 1
            time_usec = self._last_hil_sensor_time_us + 1
        self._last_hil_sensor_time_us = time_usec
        self.diagnostics.last_hil_sensor_time_usec = time_usec
        self.hil_link.mav.hil_sensor_send(
            time_usec,
            float(acceleration_frd[0]), float(acceleration_frd[1]), float(acceleration_frd[2]),
            float(gyro_frd[0]), float(gyro_frd[1]), float(gyro_frd[2]),
            *magnetic_field_frd_gauss,
            float(pressure_hpa), 0.0, float(altitude_msl_m), float(temperature_c),
            HIL_SENSOR_ALL_FIELDS,
        )
        self.diagnostics.hil_sensor_messages += 1
        if time_usec >= self._next_gps_time_us:
            self._send_hil_gps(time_usec=time_usec, position_enu_m=position, velocity_enu_mps=velocity)
            self._next_gps_time_us = time_usec + 100_000
        self.pump()
        return True

    def _send_hil_gps(self, *, time_usec: int, position_enu_m: np.ndarray, velocity_enu_mps: np.ndarray) -> None:
        if self.hil_link is None:
            return
        north_m = float(position_enu_m[0])
        east_m = -float(position_enu_m[1])
        up_m = float(position_enu_m[2])
        north_velocity = float(velocity_enu_mps[0])
        east_velocity = -float(velocity_enu_mps[1])
        up_velocity = float(velocity_enu_mps[2])
        latitude0_deg = 47.397742
        longitude0_deg = 8.545594
        earth_radius_m = 6_378_137.0
        latitude_deg = latitude0_deg + math.degrees(north_m / earth_radius_m)
        longitude_deg = longitude0_deg + math.degrees(
            east_m / (earth_radius_m * math.cos(math.radians(latitude0_deg)))
        )
        north_cmps = int(round(north_velocity * 100.0))
        east_cmps = int(round(east_velocity * 100.0))
        down_cmps = int(round(-up_velocity * 100.0))
        ground_speed_cmps = int(round(math.hypot(north_velocity, east_velocity) * 100.0))
        course_cdeg = int(round((math.degrees(math.atan2(east_velocity, north_velocity)) % 360.0) * 100.0))
        self.hil_link.mav.hil_gps_send(
            int(time_usec), 3,
            int(round(latitude_deg * 1.0e7)), int(round(longitude_deg * 1.0e7)),
            int(round((HIL_GPS_ORIGIN_ALTITUDE_MSL_M + up_m) * 1000.0)),
            60, 80, ground_speed_cmps, north_cmps, east_cmps, down_cmps,
            course_cdeg, 14,
        )
        self.diagnostics.hil_gps_messages += 1
        self.diagnostics.last_hil_gps_time_usec = int(time_usec)

    def send_gcs_heartbeat(self, *, force: bool = False) -> None:
        if self.gcs_link is None:
            return
        now = time.monotonic()
        if not force and now - self._last_gcs_heartbeat_s < 1.0:
            return
        self.gcs_link.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS,
            mavutil.mavlink.MAV_AUTOPILOT_INVALID,
            0, 0, mavutil.mavlink.MAV_STATE_ACTIVE,
        )
        self._last_gcs_heartbeat_s = now

    def send_body_velocity(self, forward_mps: float, right_mps: float, down_mps: float) -> bool:
        """Send a PX4 Offboard velocity setpoint; never sends direct thrust."""
        self.send_gcs_heartbeat()
        if self.gcs_link is None or self.target_system is None or self.target_component is None:
            self.diagnostics.offboard_send_failures += 1
            return False
        self.gcs_link.mav.set_position_target_local_ned_send(
            0, self.target_system, self.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_NED, VELOCITY_ONLY_MASK,
            0.0, 0.0, 0.0,
            float(forward_mps), float(right_mps), float(down_mps),
            0.0, 0.0, 0.0, 0.0, 0.0,
        )
        self.diagnostics.offboard_velocity_messages += 1
        self.diagnostics.last_offboard_target = (self.target_system, self.target_component)
        return True

    def send_local_velocity_ned(self, north_mps: float, east_mps: float, down_mps: float) -> bool:
        """Send a PX4-local NED velocity setpoint; never sends direct thrust."""
        self.send_gcs_heartbeat()
        if self.gcs_link is None or self.target_system is None or self.target_component is None:
            self.diagnostics.offboard_send_failures += 1
            return False
        self.gcs_link.mav.set_position_target_local_ned_send(
            0, self.target_system, self.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED, VELOCITY_ONLY_MASK,
            0.0, 0.0, 0.0,
            float(north_mps), float(east_mps), float(down_mps),
            0.0, 0.0, 0.0, 0.0, 0.0,
        )
        self.diagnostics.offboard_velocity_messages += 1
        self.diagnostics.last_offboard_target = (self.target_system, self.target_component)
        return True

    def request_offboard_mode(self) -> bool:
        """Request PX4 Offboard mode after the caller streamed setpoints."""
        if self.gcs_link is None or self.target_system is None or self.target_component is None:
            return False
        self.gcs_link.mav.command_long_send(
            self.target_system, self.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            PX4_CUSTOM_MAIN_MODE_OFFBOARD,
            0.0, 0.0, 0.0, 0.0, 0.0,
        )
        return True

    def request_arm(self) -> bool:
        """Request arming only after PX4 has accepted the Offboard mode."""
        if self.gcs_link is None or self.target_system is None or self.target_component is None:
            return False
        self.gcs_link.mav.command_long_send(
            self.target_system, self.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        )
        return True

    def request_offboard_mode_and_arm(self) -> bool:
        """Compatibility helper; callers needing acknowledgements use both methods."""
        return self.request_offboard_mode() and self.request_arm()

    def motor_wrench_world(
        self,
        *,
        world_from_body_flu: np.ndarray,
        vehicle_mass_kg: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert PX4's latest four actuator outputs to MuJoCo force/torque.

        ``motor_thrusts`` is returned for visual rotor-speed diagnostics.  No
        velocity, pose, target, Go2, QR or contact measurement enters this
        conversion: the only controller input is PX4's own HIL motor output.
        """
        controls = self.diagnostics.last_actuator_controls.copy()
        maximum_thrust_per_rotor_n = (
            float(vehicle_mass_kg) * 9.81 / (4.0 * X500_HOVER_THRUST_FRACTION)
        )
        motor_thrusts = controls * maximum_thrust_per_rotor_n
        force_body = np.array((0.0, 0.0, float(np.sum(motor_thrusts))), dtype=np.float64)
        torque_body = np.zeros(3, dtype=np.float64)
        for position, thrust, yaw_ratio in zip(
            X500_ROTOR_POSITIONS_FLU_M, motor_thrusts, X500_YAW_MOMENT_RATIO_FLU
        ):
            rotor_force = np.array((0.0, 0.0, float(thrust)), dtype=np.float64)
            torque_body += np.cross(position, rotor_force)
            torque_body[2] += float(yaw_ratio * thrust)
        rotation = np.asarray(world_from_body_flu, dtype=np.float64).reshape(3, 3)
        return rotation @ force_body, rotation @ torque_body, motor_thrusts


def rpy_ned_to_world_from_body_flu(rpy_ned_rad: np.ndarray) -> np.ndarray:
    """Convert PX4 roll/pitch/yaw (NED/FRD) to MuJoCo world-from-body FLU."""
    roll, pitch, yaw = (float(value) for value in rpy_ned_rad)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    world_ned_from_body_frd = np.array(
        (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        ),
        dtype=np.float64,
    )
    return WORLD_NWU_TO_NED @ world_ned_from_body_frd @ FLU_TO_FRD


def px4_local_ned_to_world_enu(
    *,
    start_world_enu_m: np.ndarray,
    local_ned_m: np.ndarray,
) -> np.ndarray:
    """Place PX4 EKF local NED coordinates in the MuJoCo NWU world frame.

    The historical function name is retained for the dashboard/import API,
    but this scene's horizontal Y axis is west, not east.
    """
    return np.asarray(start_world_enu_m, dtype=np.float64) + WORLD_NWU_TO_NED @ np.asarray(local_ned_m, dtype=np.float64)
