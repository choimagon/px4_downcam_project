"""MAVLink offboard velocity interface with conservative precision-landing guards."""

from __future__ import annotations

import time

from pymavlink import mavutil


class Px4MavlinkOffboard:
    PX4_CUSTOM_MAIN_MODE_OFFBOARD = 6
    VELOCITY_ONLY_MASK = 0b0000111111000111

    def __init__(self, endpoint: str = "udpin:127.0.0.1:14540") -> None:
        self.master = mavutil.mavlink_connection(endpoint, source_system=245, source_component=190)
        self.target_system: int | None = None
        self.target_component: int | None = None
        self.relative_altitude_m: float | None = None
        self._last_gcs_heartbeat = 0.0

    def connect(self, timeout_s: float = 12.0) -> None:
        heartbeat = self.master.wait_heartbeat(timeout=timeout_s)
        if heartbeat is None:
            raise TimeoutError(f"No PX4 MAVLink heartbeat on {self.master.address}")
        self.target_system = self.master.target_system
        self.target_component = self.master.target_component
        self.send_gcs_heartbeat(force=True)

    def send_gcs_heartbeat(self, force: bool = False) -> None:
        """Advertise the controller as a local GCS at 1 Hz for PX4 health checks."""
        now = time.monotonic()
        if not force and now - self._last_gcs_heartbeat < 1.0:
            return
        self.master.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS,
            mavutil.mavlink.MAV_AUTOPILOT_INVALID,
            0,
            0,
            mavutil.mavlink.MAV_STATE_ACTIVE,
        )
        self._last_gcs_heartbeat = now

    def pump(self) -> None:
        self.send_gcs_heartbeat()
        while True:
            message = self.master.recv_match(blocking=False)
            if message is None:
                return
            if message.get_type() == "GLOBAL_POSITION_INT":
                self.relative_altitude_m = float(message.relative_alt) / 1000.0
            elif message.get_type() == "LOCAL_POSITION_NED":
                self.relative_altitude_m = max(0.0, -float(message.z))

    def _send_command_and_wait(
        self, command: int, *parameters: float, timeout_s: float = 3.0
    ) -> None:
        if self.target_system is None or self.target_component is None:
            raise RuntimeError("Call connect() before sending a PX4 command")
        padded = list(parameters[:7]) + [0.0] * (7 - len(parameters))
        self.master.mav.command_long_send(
            self.target_system, self.target_component, command, 0, *padded
        )
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            message = self.master.recv_match(type="COMMAND_ACK", blocking=True, timeout=0.5)
            if message is None or int(message.command) != command:
                continue
            if int(message.result) == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                return
            raise RuntimeError(f"PX4 rejected MAVLink command {command}: result={message.result}")
        raise TimeoutError(f"PX4 did not acknowledge MAVLink command {command}")

    def send_body_velocity(self, forward_mps: float, right_mps: float, down_mps: float) -> None:
        if self.target_system is None or self.target_component is None:
            raise RuntimeError("Call connect() before sending offboard setpoints")
        self.master.mav.set_position_target_local_ned_send(
            0,
            self.target_system,
            self.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_NED,
            self.VELOCITY_ONLY_MASK,
            0.0,
            0.0,
            0.0,
            float(forward_mps),
            float(right_mps),
            float(down_mps),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

    def enable_offboard_and_arm(self) -> None:
        if self.target_system is None or self.target_component is None:
            raise RuntimeError("Call connect() before enabling offboard")
        # PX4 requires a stream of setpoints before switching to Offboard.
        for _ in range(20):
            self.send_body_velocity(0.0, 0.0, 0.0)
            time.sleep(0.1)
        self._send_command_and_wait(
            mavutil.mavlink.MAV_CMD_DO_SET_MODE,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            self.PX4_CUSTOM_MAIN_MODE_OFFBOARD,
            0.0,
        )
        self._send_command_and_wait(
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            1.0,
        )

    def land(self) -> None:
        if self.target_system is None or self.target_component is None:
            return
        self._send_command_and_wait(mavutil.mavlink.MAV_CMD_NAV_LAND)

    def close(self) -> None:
        self.master.close()
