"""
franka_robot.py
---------------
Reads Franka FR3 robot state from the UDP JSON stream broadcast by
franka_vr_control.cpp (port 9091, loopback) and exposes convenient
Python accessors that match the original FrankaRobot interface.

State JSON fields published by franka_vr_control.cpp:
  robot0_joint_pos        – q[0..6]                    (7,) rad
  robot0_joint_vel        – dq[0..6]                   (7,) rad/s
  robot0_eef_pos          – O_T_EE translation          (3,) m
  robot0_eef_quat         – O_T_EE rotation [qx,qy,qz,qw]  (4,)
  robot0_gripper_qpos     – actual gripper width        scalar m
  robot0_joint_ext_torque – tau_ext_hat_filtered        (7,) N·m
  robot0_force_ee         – O_F_ext_hat_K[0:3]          (3,) N
  robot0_torque_ee        – O_F_ext_hat_K[3:6]          (3,) N·m

Command channel (port 8888) sends VR pose packets directly to
franka_vr_control.cpp – the old franka_r3_robot_server command channel
(port 8889) is no longer used and has been removed.
"""

import json
import os
import socket
import struct
import threading
import time

import numpy as np
from dataclasses import dataclass, field
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# RobotInputs – action / velocity command dataclass (interface unchanged)
# ---------------------------------------------------------------------------

@dataclass
class RobotInputs:
    """Action vector [vx, vy, vz, wx, wy, wz, gripper] plus recording flags."""

    left_x:        float
    left_y:        float
    right_z:       float
    roll:          float
    pitch:         float
    yaw:           float
    gripper:       float
    square_btn:    bool   # unused – kept for interface compatibility
    triangle_btn:  bool   # unused – kept for interface compatibility

    def __init__(self, array=None, square_btn=None, triangle_btn=None):
        if array is not None:
            self.left_x       = array[0]
            self.left_y       = array[1]
            self.right_z      = array[2]
            self.roll         = array[3]
            self.pitch        = array[4]
            self.yaw          = array[5]
            self.gripper      = array[6]
            self.square_btn   = square_btn   if square_btn   is not None else False
            self.triangle_btn = triangle_btn if triangle_btn is not None else False
        else:
            self.left_x       = 0.0
            self.left_y       = 0.0
            self.right_z      = 0.0
            self.roll         = 0.0
            self.pitch        = 0.0
            self.yaw          = 0.0
            self.gripper      = 0.0
            self.square_btn   = False
            self.triangle_btn = False


# ---------------------------------------------------------------------------
# FrankaRobot – reads state broadcast by franka_vr_control.cpp
# ---------------------------------------------------------------------------

class FrankaRobot:
    """
    Listens for the JSON robot-state UDP packets that franka_vr_control.cpp
    broadcasts on STATE_PORT (9091) and exposes them through the same
    accessor API as the original class.

    Sending motion commands
    -----------------------
    franka_vr_control.cpp is the sole motion controller. To command the
    robot you must send a VR pose packet to it on port 8888 in the format:

        "pos_x pos_y pos_z quat_x quat_y quat_z quat_w button speed force eps_i eps_o"

    A convenience helper ``send_vr_pose()`` is provided below.

    Usage
    -----
        robot = FrankaRobot()
        q   = robot.get_joint_positions()   # np.ndarray (7,)
        pos = robot.get_ee_translation()     # np.ndarray (3,)
        robot.close()
    """

    # ---- ports ----------------------------------------------------------------
    STATE_PORT    = 9091          # matches C++ STATE_PORT
    VR_CMD_PORT   = 8888          # franka_vr_control_data_sender_client (VR teleoperation)
    VR_CMD_IP     = "127.0.0.1"
    CMD_PORT      = 8889          # franka_r3_robot_server (replay / direct Python control)
    CMD_IP        = "127.0.0.1"

    # ---- home joint configuration --------------------------------------------
    HOME_JOINTS = [0.0, -0.48, 0.0, -2.0, 0.0, 1.57, -0.85]  # [rad]

    # ---- DH parameters for Jacobian (standard Panda / FR3) -------------------
    _DH_A     = np.array([0.0, 0.0, 0.0, 0.0825, -0.0825, 0.0,  0.088, 0.0])
    _DH_D     = np.array([0.333, 0.0, 0.316, 0.0, 0.384, 0.0, 0.0, 0.107])
    _DH_ALPHA = np.array([
        0.0, -np.pi/2, np.pi/2, np.pi/2, -np.pi/2, np.pi/2, np.pi/2, 0.0
    ])

    def __init__(
        self,
        state_port:       int   = STATE_PORT,
        state_listen_ip:  str   = "0.0.0.0",
        vr_cmd_port:      int   = VR_CMD_PORT,
        vr_cmd_ip:        str   = VR_CMD_IP,
        cmd_port:         int   = CMD_PORT,
        cmd_ip:           str   = CMD_IP,
        connect_timeout:  float = 10.0,
    ):
        self._state_port   = int(state_port)
        self._vr_cmd_ip    = vr_cmd_ip
        self._vr_cmd_port  = vr_cmd_port
        self._cmd_ip       = cmd_ip
        self._cmd_port     = int(cmd_port)
        self._debug_state  = os.environ.get("COLLECT_DATA_DEBUG_STATE", "0") == "1"

        # ---- state receiver ---------------------------------------------------
        self._lock              = threading.Lock()
        self._latest_state      = None
        self._rx_count          = 0
        self._last_rx_time      = None
        self._first_packet_logged = False
        self._running           = True

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        self._sock.bind((state_listen_ip, self._state_port))
        self._sock.settimeout(0.2)

        self._thread = threading.Thread(
            target=self._receiver_loop,
            daemon=True,
            name="franka-state-rx",
        )
        self._thread.start()

        # Wait for the first state packet so callers get valid data immediately
        deadline = time.monotonic() + connect_timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._latest_state is not None:
                    break
            time.sleep(0.05)
        else:
            print(
                f"[FrankaRobot] Warning: no state received within "
                f"{connect_timeout}s on port {state_port}. "
                "Is franka_vr_control.cpp running?"
            )

    # ---------------------------------------------------------------------------
    # Internal – state receiver loop
    # ---------------------------------------------------------------------------

    def _receiver_loop(self):
        while self._running:
            try:
                data, _ = self._sock.recvfrom(1 << 16)
                text = data.decode("utf-8", errors="ignore").strip()
                if not text:
                    continue
                # Take the last complete JSON line (handles buffered multi-line packets)
                line = text.splitlines()[-1]
                state = json.loads(line)
                with self._lock:
                    self._latest_state = state
                    self._rx_count    += 1
                    self._last_rx_time = time.monotonic()
                    if self._debug_state and not self._first_packet_logged:
                        print(
                            f"[DEBUG_STATE] first packet on UDP {self._state_port}: "
                            f"keys={sorted(state.keys())}"
                        )
                        print(
                            f"[DEBUG_STATE] joint_pos={state.get('robot0_joint_pos')}  "
                            f"eef_pos={state.get('robot0_eef_pos')}"
                        )
                        self._first_packet_logged = True
            except socket.timeout:
                continue
            except Exception as exc:
                if self._running:
                    print(f"[FrankaRobot] state receiver error: {exc}")
                time.sleep(0.02)

    def _snapshot(self) -> dict | None:
        """Thread-safe copy of the latest state dict."""
        with self._lock:
            return dict(self._latest_state) if self._latest_state is not None else None

    # ---------------------------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------------------------

    def get_state_diagnostics(self) -> dict:
        """Health check for the UDP state stream."""
        required_keys = (
            "robot0_joint_pos",
            "robot0_joint_vel",
            "robot0_eef_pos",
            "robot0_eef_quat",
        )
        with self._lock:
            state        = dict(self._latest_state) if self._latest_state is not None else None
            rx_count     = self._rx_count
            last_rx_time = self._last_rx_time

        missing_keys = list(required_keys) if state is None else [
            k for k in required_keys if k not in state
        ]
        last_rx_age_s = None
        if last_rx_time is not None:
            last_rx_age_s = max(0.0, time.monotonic() - last_rx_time)

        return {
            "state_port":            self._state_port,
            "has_state":             state is not None,
            "rx_count":              rx_count,
            "last_rx_age_s":         last_rx_age_s,
            "missing_required_keys": missing_keys,
        }

    def has_valid_state(self) -> bool:
        diag = self.get_state_diagnostics()
        return diag["has_state"] and len(diag["missing_required_keys"]) == 0

    def close(self):
        self._running = False
        try:
            self._sock.close()
        except Exception:
            pass

    # ---------------------------------------------------------------------------
    # Command channel – send VR pose packet to franka_vr_control.cpp
    # ---------------------------------------------------------------------------

    def send_vr_pose(
        self,
        pos_x: float, pos_y: float, pos_z: float,
        quat_x: float, quat_y: float, quat_z: float, quat_w: float,
        button_pressed: float = 0.0,
        gripper_speed:  float = 0.1,
        gripper_force:  float = 20.0,
        epsilon_inner:  float = 0.005,
        epsilon_outer:  float = 0.005,
    ) -> None:
        """
        Send a VR pose command to franka_vr_control.cpp (non-blocking).

        The packet format is the space-separated 12-field string that the C++
        networkThread expects:
          pos_x pos_y pos_z  quat_x quat_y quat_z quat_w
          button_pressed gripper_speed gripper_force epsilon_inner epsilon_outer
        """
        msg = (
            f"{pos_x:.6f} {pos_y:.6f} {pos_z:.6f} "
            f"{quat_x:.6f} {quat_y:.6f} {quat_z:.6f} {quat_w:.6f} "
            f"{button_pressed:.1f} {gripper_speed:.4f} "
            f"{gripper_force:.4f} {epsilon_inner:.6f} {epsilon_outer:.6f}"
        )
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.sendto(msg.encode(), (self._vr_cmd_ip, self._vr_cmd_port))

    def send_gripper_toggle(
        self,
        gripper_speed:  float = 0.1,
        gripper_force:  float = 20.0,
        epsilon_inner:  float = 0.005,
        epsilon_outer:  float = 0.005,
    ) -> None:
        """
        Simulate a button press to toggle the gripper.
        Uses the current EE pose so the robot does not move.
        """
        pos = self.get_ee_translation()
        q   = self.get_ee_quaternion()   # [qx, qy, qz, qw]
        self.send_vr_pose(
            pos[0], pos[1], pos[2],
            q[0], q[1], q[2], q[3],
            button_pressed=1.0,
            gripper_speed=gripper_speed,
            gripper_force=gripper_force,
            epsilon_inner=epsilon_inner,
            epsilon_outer=epsilon_outer,
        )

    # ---------------------------------------------------------------------------
    # Command channel – franka_r3_robot_server (replay / direct Python control)
    # Run: ./franka_r3_robot_server <robot-ip> 127.0.0.1 9091
    # ---------------------------------------------------------------------------

    def _send_cmd(self, cmd: str, wait_ack: bool = True, timeout: float = 30.0) -> bool:
        """
        Send a text command to franka_r3_robot_server (port 8889) and
        optionally block until the server replies "done".

        The server's command thread is strictly sequential – it blocks on
        waitForMotionDone() before reading the next packet.  Always use
        wait_ack=True for motion commands so we don't flood the socket buffer.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.sendto(cmd.encode(), (self._cmd_ip, self._cmd_port))
            if not wait_ack:
                return True
            s.settimeout(timeout)
            try:
                data, _ = s.recvfrom(256)
                ack = data.decode().strip()
                if ack != "done":
                    print(f"[FrankaRobot] Unexpected ack: {ack!r} for cmd: {cmd!r}")
                return ack == "done"
            except socket.timeout:
                print(f"[FrankaRobot] Command ack timed out after {timeout}s: {cmd!r}")
                return False

    def move_home(self, timeout: float = 30.0) -> None:
        """
        Move to the home joint configuration via franka_r3_robot_server.
        Blocks until the server signals motion complete (or timeout expires).
        """
        joints_str = " ".join(f"{q:.6f}" for q in self.HOME_JOINTS)
        print("[FrankaRobot] Moving to home position...")
        ok = self._send_cmd(f"joint {joints_str}", wait_ack=True, timeout=timeout)
        if ok:
            print("[FrankaRobot] Home position reached.")
        else:
            print("[FrankaRobot] Warning: home ack timed out — robot may not be at home.")

    def send_joints(self, q, wait_ack: bool = True, timeout: float = 30.0) -> bool:
        """
        Send a joint position target to franka_r3_robot_server and wait for
        the server to finish the motion before returning (wait_ack=True by default).
        q: array-like of 7 joint angles [rad].
        Returns True if the server acknowledged "done", False on timeout.
        """
        joints_str = " ".join(f"{float(v):.6f}" for v in q)
        return self._send_cmd(f"joint {joints_str}", wait_ack=wait_ack, timeout=timeout)

    def move_velocity_array(
        self,
        linear_velocity,
        angular_velocity,
        gripper: float,
        duration_ms: float = 100.0,
    ) -> None:
        """
        Integrate a Cartesian velocity command into a pose target and send it
        to franka_r3_robot_server as a "pose x y z qx qy qz qw" command.

        Blocks until the server finishes executing the motion (wait_ack=True),
        which keeps the command stream synchronised with the robot.

        Args:
            linear_velocity:  (3,) array-like, [m/s] in base frame
            angular_velocity: (3,) array-like, [rad/s] axis-angle in base frame
            gripper:          >0.5 → open, <-0.5 → close, else no change
            duration_ms:      integration window [ms]; should match recording dt
        """
        dt  = duration_ms / 1000.0
        lv  = np.asarray(linear_velocity,  dtype=float)
        av  = np.asarray(angular_velocity, dtype=float)

        curr_pos = self.get_ee_translation()          # (3,) m
        curr_q   = self.get_ee_quaternion()           # [qx, qy, qz, qw]

        new_pos = curr_pos + lv * dt

        # Integrate angular velocity: delta_rot = exp(av * dt) applied left
        delta_rot = Rotation.from_rotvec(av * dt)
        new_q     = (delta_rot * Rotation.from_quat(curr_q)).as_quat()  # [qx,qy,qz,qw]

        cmd = (
            f"pose {new_pos[0]:.6f} {new_pos[1]:.6f} {new_pos[2]:.6f} "
            f"{new_q[0]:.6f} {new_q[1]:.6f} {new_q[2]:.6f} {new_q[3]:.6f}"
        )
        # Fire-and-forget: the server stores the new Ruckig target lock-free and
        # blends into it continuously, exactly like the VR teleoperation loop.
        # Waiting for "done" would stall replay because the command thread blocks
        # until Ruckig finishes the full trajectory for each tiny increment.
        self._send_cmd(cmd, wait_ack=False)

        if gripper > 0.5:
            self._send_cmd("gripper_open 0.1", wait_ack=False)
        elif gripper < -0.5:
            self._send_cmd("gripper_close 0.0 0.1 50.0 0.03 0.02", wait_ack=False)

    # ---------------------------------------------------------------------------
    # State accessors  (keys match JSON broadcast by franka_vr_control.cpp)
    # ---------------------------------------------------------------------------

    def get_joint_positions(self) -> np.ndarray:
        """Joint positions q[0..6] (7,) [rad]."""
        s = self._snapshot()
        if s and "robot0_joint_pos" in s:
            return np.asarray(s["robot0_joint_pos"], dtype=float).reshape(7)
        return np.zeros(7)

    def get_joint_velocities(self) -> np.ndarray:
        """Joint velocities dq[0..6] (7,) [rad/s]."""
        s = self._snapshot()
        if s and "robot0_joint_vel" in s:
            return np.asarray(s["robot0_joint_vel"], dtype=float).reshape(7)
        return np.zeros(7)

    def get_ee_translation(self) -> np.ndarray:
        """End-effector position (3,) [m] in robot base frame."""
        s = self._snapshot()
        if s and "robot0_eef_pos" in s:
            return np.asarray(s["robot0_eef_pos"], dtype=float).reshape(3)
        return np.zeros(3)

    def get_ee_quaternion(self) -> np.ndarray:
        """End-effector orientation [qx, qy, qz, qw] (4,) – scipy convention."""
        s = self._snapshot()
        if s and "robot0_eef_quat" in s:
            q = np.asarray(s["robot0_eef_quat"], dtype=float).reshape(4)
            n = np.linalg.norm(q)
            return q / n if n > 1e-12 else np.array([0.0, 0.0, 0.0, 1.0])
        return np.array([0.0, 0.0, 0.0, 1.0])

    def get_gripper_width(self) -> float:
        """
        Actual gripper finger separation [m].
        Populated from franka::GripperState::width read by the C++ worker thread.
        """
        s = self._snapshot()
        if s and "robot0_gripper_qpos" in s:
            return float(s["robot0_gripper_qpos"])
        return 0.0

    def get_tau_J(self) -> np.ndarray:
        """
        External joint torques tau_ext_hat_filtered (7,) [N·m].
        Mapped from robot0_joint_ext_torque in the state packet.
        """
        s = self._snapshot()
        if s and "robot0_joint_ext_torque" in s:
            return np.asarray(s["robot0_joint_ext_torque"], dtype=float).reshape(7)
        return np.zeros(7)

    def get_tau_ext_hat_filtered(self) -> np.ndarray:
        """Alias for get_tau_J() – same underlying data."""
        return self.get_tau_J()

    def get_O_F_ext_hat_K(self) -> np.ndarray:
        """
        External wrench at EE in base frame (6,) [N, N, N, N·m, N·m, N·m].
        Concatenated from robot0_force_ee and robot0_torque_ee.
        """
        s = self._snapshot()
        if s and "robot0_force_ee" in s and "robot0_torque_ee" in s:
            f = np.asarray(s["robot0_force_ee"],  dtype=float).reshape(3)
            t = np.asarray(s["robot0_torque_ee"], dtype=float).reshape(3)
            return np.concatenate([f, t])
        return np.zeros(6)

    # Fields not streamed – return safe zero values

    def get_elbow_joint3_pos(self) -> float:
        """Not available in UDP stream; returns 0.0."""
        return 0.0

    def get_elbow_joint4_flip(self) -> float:
        """Not available in UDP stream; returns 0.0."""
        return 0.0

    def get_dtau_J(self) -> np.ndarray:
        """Not available in UDP stream; returns zeros (7,)."""
        return np.zeros(7)

    # ---------------------------------------------------------------------------
    # Derived quantities
    # ---------------------------------------------------------------------------

    @staticmethod
    def quaternion_array_to_rpy(quaternion, degrees: bool = False) -> np.ndarray:
        """Convert [qx, qy, qz, qw] → [roll, pitch, yaw]."""
        return Rotation.from_quat(quaternion).as_euler("xyz", degrees=degrees)

    @staticmethod
    def _dh_transform(a: float, d: float, alpha: float, theta: float) -> np.ndarray:
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        return np.array([
            [ct, -st*ca,  st*sa, a*ct],
            [st,  ct*ca, -ct*sa, a*st],
            [0.0,    sa,     ca,    d],
            [0.0,   0.0,    0.0,  1.0],
        ])

    def _compute_jacobian(self, q: np.ndarray) -> np.ndarray:
        """Geometric Jacobian (6×7) at configuration q."""
        T = np.eye(4)
        T_chain = [T.copy()]
        for i in range(7):
            T = T @ self._dh_transform(
                self._DH_A[i], self._DH_D[i], self._DH_ALPHA[i], q[i]
            )
            T_chain.append(T.copy())
        # Fixed EE transform
        T = T @ self._dh_transform(self._DH_A[7], self._DH_D[7], self._DH_ALPHA[7], 0.0)

        o_n = T[:3, 3]
        J   = np.zeros((6, 7))
        for i in range(7):
            z_i = T_chain[i][:3, 2]
            o_i = T_chain[i][:3, 3]
            J[:3, i] = np.cross(z_i, o_n - o_i)
            J[3:, i] = z_i
        return J

    def get_ee_twist_linear(self) -> np.ndarray:
        """EE linear velocity (3,) [m/s] via J(q)·dq."""
        q  = self.get_joint_positions()
        dq = self.get_joint_velocities()
        if not (np.isfinite(q).all() and np.isfinite(dq).all()):
            return np.zeros(3)
        return self._compute_jacobian(q)[:3] @ dq

    def get_ee_twist_angular(self) -> np.ndarray:
        """EE angular velocity (3,) [rad/s] via J(q)·dq."""
        q  = self.get_joint_positions()
        dq = self.get_joint_velocities()
        if not (np.isfinite(q).all() and np.isfinite(dq).all()):
            return np.zeros(3)
        return self._compute_jacobian(q)[3:] @ dq

    def get_ee_state(self) -> dict:
        """Convenience bundle of common EE state fields."""
        q = self.get_ee_quaternion()
        return {
            "ee_pos_t":   self.get_ee_translation(),
            "ee_pos_rpy": self.quaternion_array_to_rpy(q),
            "gpos":       np.array([self.get_gripper_width()]),
        }


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Connecting to franka_vr_control.cpp state stream …")
    robot = FrankaRobot()

    diag = robot.get_state_diagnostics()
    print(f"Diagnostics: {diag}")

    if robot.has_valid_state():
        print(f"Joint positions : {robot.get_joint_positions()}")
        print(f"EE translation  : {robot.get_ee_translation()}")
        print(f"EE quaternion   : {robot.get_ee_quaternion()}")
        print(f"Gripper width   : {robot.get_gripper_width():.4f} m")
        print(f"Ext. torques    : {robot.get_tau_J()}")
        print(f"Ext. wrench     : {robot.get_O_F_ext_hat_K()}")
        print(f"EE linear vel   : {robot.get_ee_twist_linear()}")
        ee_state = robot.get_ee_state()
        print(f"EE RPY          : {ee_state['ee_pos_rpy']}")
    else:
        print("No valid state yet – is franka_vr_control running?")

    robot.close()