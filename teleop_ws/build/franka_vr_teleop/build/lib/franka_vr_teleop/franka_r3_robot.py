#!/usr/bin/env python3
"""
FR3 UDP client (non-VR protocol).

Command protocol (to franka_pose_cmd_client):
  "<x> <y> <z> <qx> <qy> <qz> <qw>"

State protocol (from franka_pose_cmd_client, JSON over UDP):
  robot0_eef_pos, robot0_eef_quat, robot0_joint_pos, robot0_joint_vel,
  robot0_joint_ext_torque, timestamp
"""

from __future__ import annotations

import json
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


@dataclass
class EEFPose:
    """Simple pose container compatible with franky-like access."""

    translation: np.ndarray
    quaternion: np.ndarray


class FR3_ROBOT:
    """
    Robot wrapper with FR3_ROBOT-like API, using a plain pose command client.

    Defaults target `franka_pose_cmd_client`:
      - command UDP port: 8890
      - state UDP port:   9093
    """

    CMD_PORT = 8890
    STATE_PORT = 9093
    DEFAULT_CONTROL_RATE = 100.0

    # Standard Panda/FR3 DH chain (7 joints + static tool transform)
    _DH_A = np.array([0.0, 0.0, 0.0, 0.0825, -0.0825, 0.0, 0.088, 0.0], dtype=float)
    _DH_D = np.array([0.333, 0.0, 0.316, 0.0, 0.384, 0.0, 0.0, 0.107], dtype=float)
    _DH_ALPHA = np.array(
        [0.0, -np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2, np.pi / 2, 0.0],
        dtype=float,
    )

    def __init__(
        self,
        ip_address: str,
        cmd_port: int = CMD_PORT,
        state_listen_ip: str = "0.0.0.0",
        state_port: Optional[int] = STATE_PORT,
        connect_timeout: float = 2.0,
        control_rate: float = DEFAULT_CONTROL_RATE,
    ):
        self.ip_address = ip_address
        self.cmd_port = int(cmd_port)
        self.control_rate = float(control_rate)

        # Gripper is not controlled by this non-VR pose client.
        self.grasp_width = 0.02
        self._gripper_open_estimate = True

        self._cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._cmd_lock = threading.Lock()

        self._state_lock = threading.Lock()
        self._latest_state: Optional[Dict[str, object]] = None
        self._warned_keys = set()
        self._running = True
        self._state_thread: Optional[threading.Thread] = None
        self._state_sock: Optional[socket.socket] = None

        self._command_position = np.zeros(3, dtype=float)
        self._command_orientation = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)

        self._async_motion_thread: Optional[threading.Thread] = None
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="fr3-client")

        # Metadata not provided by pose client stream.
        self._F_T_EE = np.eye(4, dtype=float)
        self._EE_T_K = np.eye(4, dtype=float)
        self._configured_mass = 0.0
        self._configured_inertia = np.eye(3, dtype=float)
        self._configured_com = np.zeros(3, dtype=float)

        if state_port is not None:
            self._state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._state_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
            self._state_sock.bind((state_listen_ip, int(state_port)))
            self._state_sock.settimeout(0.2)

            self._state_thread = threading.Thread(
                target=self._state_receiver_loop,
                name="fr3-state-rx",
                daemon=True,
            )
            self._state_thread.start()

            if connect_timeout > 0.0:
                deadline = time.monotonic() + float(connect_timeout)
                while time.monotonic() < deadline:
                    if self._get_state_snapshot() is not None:
                        break
                    time.sleep(0.02)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self):
        self._running = False
        if self._state_thread is not None and self._state_thread.is_alive():
            self._state_thread.join(timeout=1.0)
        try:
            self._cmd_sock.close()
        except Exception:
            pass
        if self._state_sock is not None:
            try:
                self._state_sock.close()
            except Exception:
                pass
        self._executor.shutdown(wait=False, cancel_futures=True)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_quat(quat: Sequence[float]) -> np.ndarray:
        q = np.asarray(quat, dtype=float).reshape(4)
        n = np.linalg.norm(q)
        if n < 1e-12:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        return q / n

    @staticmethod
    def _build_T(position: Sequence[float], quaternion: Sequence[float]) -> np.ndarray:
        T = np.eye(4, dtype=float)
        T[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
        T[:3, 3] = np.asarray(position, dtype=float).reshape(3)
        return T

    def _warn_once(self, key: str, message: str):
        if key in self._warned_keys:
            return
        self._warned_keys.add(key)
        print(message)

    def _state_receiver_loop(self):
        assert self._state_sock is not None
        while self._running:
            try:
                data, _ = self._state_sock.recvfrom(1 << 16)
                text = data.decode("utf-8", errors="ignore").strip()
                if not text:
                    continue
                line = text.splitlines()[-1]
                state = json.loads(line)
                with self._state_lock:
                    self._latest_state = state
            except socket.timeout:
                continue
            except Exception as exc:
                if self._running:
                    self._warn_once("state_rx_error", f"[FR3_ROBOT] state receiver error: {exc}")
                time.sleep(0.02)

    def _get_state_snapshot(self) -> Optional[Dict[str, object]]:
        with self._state_lock:
            if self._latest_state is None:
                return None
            return dict(self._latest_state)

    def _get_current_pose_for_command(self) -> Tuple[np.ndarray, np.ndarray]:
        state = self._get_state_snapshot()
        if state is not None and "robot0_eef_pos" in state and "robot0_eef_quat" in state:
            pos = np.asarray(state["robot0_eef_pos"], dtype=float).reshape(3)
            quat = self._normalize_quat(state["robot0_eef_quat"])
            return pos, quat
        return self._command_position.copy(), self._command_orientation.copy()

    def _send_pose_command(self, position: Sequence[float], orientation: Sequence[float]):
        pos = np.asarray(position, dtype=float).reshape(3)
        quat = self._normalize_quat(orientation)
        message = (
            f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} "
            f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}"
        )
        with self._cmd_lock:
            self._cmd_sock.sendto(message.encode("utf-8"), (self.ip_address, self.cmd_port))
            self._command_position = pos
            self._command_orientation = quat

    def send_eef_pose_command(self, position: Sequence[float], orientation: Sequence[float]):
        """Public direct command entry point for absolute end-effector pose."""
        self._send_pose_command(position, orientation)

    @staticmethod
    def _dh_transform(a: float, d: float, alpha: float, theta: float) -> np.ndarray:
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        return np.array(
            [
                [ct, -st * ca, st * sa, a * ct],
                [st, ct * ca, -ct * sa, a * st],
                [0.0, sa, ca, d],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

    def _fk_with_chain(self, joints: Sequence[float]) -> Tuple[np.ndarray, list]:
        q = np.asarray(joints, dtype=float).reshape(7)
        T = np.eye(4, dtype=float)
        T_chain = [T.copy()]
        for i in range(7):
            T = T @ self._dh_transform(self._DH_A[i], self._DH_D[i], self._DH_ALPHA[i], q[i])
            T_chain.append(T.copy())
        T = T @ self._dh_transform(self._DH_A[7], self._DH_D[7], self._DH_ALPHA[7], 0.0)
        return T, T_chain

    # ------------------------------------------------------------------
    # State retrieval
    # ------------------------------------------------------------------

    def get_jacobian_at_current_state(self):
        q = self.get_joint_positions()
        if q.shape[0] != 7 or not np.isfinite(q).all():
            self._warn_once("jacobian_no_state", "[FR3_ROBOT] Joint state unavailable; returning zero Jacobian.")
            return np.zeros((6, 7), dtype=float)

        T_ee, T_chain = self._fk_with_chain(q)
        o_n = T_ee[:3, 3]
        J = np.zeros((6, 7), dtype=float)
        for i in range(7):
            z = T_chain[i][:3, 2]
            o = T_chain[i][:3, 3]
            J[:3, i] = np.cross(z, o_n - o)
            J[3:, i] = z
        return J

    def get_wrench_at_ee_via_jacobian(self):
        J = self.get_jacobian_at_current_state()
        tau = self.get_joint_torques()
        if tau.shape[0] != 7 or not np.isfinite(tau).all():
            return np.zeros(6, dtype=float)
        return np.linalg.pinv(J.T) @ tau

    def get_joint_positions(self):
        state = self._get_state_snapshot()
        if state is None or "robot0_joint_pos" not in state:
            self._warn_once("joint_pos_missing", "[FR3_ROBOT] No state stream (joint positions); returning NaNs.")
            return np.full(7, np.nan, dtype=float)
        return np.asarray(state["robot0_joint_pos"], dtype=float).reshape(7)

    def get_joint_torques(self):
        state = self._get_state_snapshot()
        if state is None or "robot0_joint_ext_torque" not in state:
            self._warn_once("joint_tau_missing", "[FR3_ROBOT] No state stream (joint torques); returning NaNs.")
            return np.full(7, np.nan, dtype=float)
        return np.asarray(state["robot0_joint_ext_torque"], dtype=float).reshape(7)

    def get_end_effector_wrench(self):
        return self.get_wrench_at_ee_via_jacobian()

    def get_wrench_at_base_frame(self):
        wrench_ee = self.get_end_effector_wrench()
        if wrench_ee.shape[0] != 6 or not np.isfinite(wrench_ee).all():
            return np.zeros(6, dtype=float)
        rot = self.get_O_T_EE()[:3, :3]
        force_base = rot @ wrench_ee[:3]
        torque_base = rot @ wrench_ee[3:]
        return np.concatenate([force_base, torque_base])

    def get_ee_pose_in_flange_frame(self):
        return self._F_T_EE.copy()

    def get_joint_velocities(self):
        state = self._get_state_snapshot()
        if state is None or "robot0_joint_vel" not in state:
            self._warn_once("joint_vel_missing", "[FR3_ROBOT] No state stream (joint velocities); returning NaNs.")
            return np.full(7, np.nan, dtype=float)
        return np.asarray(state["robot0_joint_vel"], dtype=float).reshape(7)

    def get_joint_states(self):
        return {
            "position": self.get_joint_positions(),
            "velocity": self.get_joint_velocities(),
            "effort": self.get_joint_torques(),
        }

    def get_end_effector_pose(self):
        state = self._get_state_snapshot()
        if state is not None and "robot0_eef_pos" in state and "robot0_eef_quat" in state:
            pos = np.asarray(state["robot0_eef_pos"], dtype=float).reshape(3)
            quat = self._normalize_quat(state["robot0_eef_quat"])
            return EEFPose(translation=pos, quaternion=quat)
        return EEFPose(
            translation=self._command_position.copy(),
            quaternion=self._command_orientation.copy(),
        )

    def get_eef_position(self):
        return self.get_end_effector_pose().translation

    def get_eef_orientation(self):
        return self.get_end_effector_pose().quaternion

    def get_O_T_EE(self):
        pose = self.get_end_effector_pose()
        return self._build_T(pose.translation, pose.quaternion)

    def get_EE_T_K(self):
        return self._EE_T_K.copy()

    def get_configured_mass_in_EE_frame(self):
        return float(self._configured_mass)

    def get_configured_inertia_rotation_matrix_in_EE_frame(self):
        return self._configured_inertia.copy()

    def get_CoM_in_flange_frame(self):
        return self._configured_com.copy()

    # ------------------------------------------------------------------
    # Motion control
    # ------------------------------------------------------------------

    @staticmethod
    def generate_smooth_path(start_pos, start_quat, target_pos, target_quat, steps=10):
        start_pos = np.asarray(start_pos, dtype=float).reshape(3)
        target_pos = np.asarray(target_pos, dtype=float).reshape(3)
        start_quat = FR3_ROBOT._normalize_quat(start_quat)
        target_quat = FR3_ROBOT._normalize_quat(target_quat)

        key_rots = Rotation.from_quat([start_quat, target_quat])
        slerp = Slerp([0.0, 1.0], key_rots)

        waypoints = []
        for i in range(1, int(steps) + 1):
            t = i / float(steps)
            alpha = 0.5 * (1.0 - np.cos(np.pi * t))
            interp_pos = (1.0 - alpha) * start_pos + alpha * target_pos
            interp_quat = slerp([alpha]).as_quat()[0]
            waypoints.append((interp_pos, interp_quat))
        return waypoints

    def _is_pose_reached(self, target_pos, target_quat, pos_tol=0.005, rot_tol_deg=3.0):
        current_pos = self.get_eef_position()
        current_quat = self._normalize_quat(self.get_eef_orientation())
        target_pos = np.asarray(target_pos, dtype=float).reshape(3)
        target_quat = self._normalize_quat(target_quat)

        pos_err = np.linalg.norm(current_pos - target_pos)
        dot = np.clip(np.abs(np.dot(current_quat, target_quat)), 0.0, 1.0)
        rot_err_deg = np.degrees(2.0 * np.arccos(dot))
        reached = pos_err <= float(pos_tol) and rot_err_deg <= float(rot_tol_deg)
        return reached, pos_err, rot_err_deg

    def _sleep_tick(self):
        if self.control_rate > 0.0:
            time.sleep(1.0 / self.control_rate)

    def move_to_pose(self, position, orientation, steps=0):
        start_pos = self.get_eef_position()
        start_quat = self.get_eef_orientation()

        target_pos = np.asarray(position, dtype=float).reshape(3)
        target_quat = self._normalize_quat(orientation)
        if steps is None or int(steps) <= 0:
            steps = 15
        steps = max(1, int(steps))

        path = self.generate_smooth_path(start_pos, start_quat, target_pos, target_quat, steps=steps)
        for interp_pos, interp_quat in path:
            self._send_pose_command(interp_pos, interp_quat)
            self._sleep_tick()

        reached, pos_err, rot_err = self._is_pose_reached(target_pos, target_quat)
        if not reached:
            print(
                f"[move_to_pose] Warning: target not reached - "
                f"pos_err={pos_err * 1000.0:.1f}mm, rot_err={rot_err:.1f}deg"
            )

    def move_to_pose_for_vla(self, rel_position, rel_orientation):
        current_pos = self.get_eef_position()
        current_quat = self.get_eef_orientation()
        target_pos = current_pos + np.asarray(rel_position, dtype=float).reshape(3)
        target_rot = Rotation.from_quat(current_quat) * Rotation.from_quat(rel_orientation)
        target_quat = target_rot.as_quat()
        self.move_to_pose(target_pos, target_quat, steps=15)

    def move_and_grasp_for_vla(self, vla_action):
        rel_position = np.asarray(vla_action[:3], dtype=float)
        rel_orientation = np.asarray(vla_action[3:6], dtype=float)
        rel_quat = Rotation.from_rotvec(rel_orientation).as_quat()
        self.move_to_pose_for_vla(rel_position, rel_quat)

    def relative_move_and_grasp_for_vla(self, vla_action):
        rel_position = np.asarray(vla_action[:3], dtype=float)
        rel_rotvec = np.asarray(vla_action[3:6], dtype=float)
        self.relative_move_to_pose(rel_position, relative_rot_vec=rel_rotvec)

    def relative_move_to_pose(self, relative_position, relative_rot_vec):
        relative_position = np.asarray(relative_position, dtype=float).reshape(3)
        relative_quat = Rotation.from_rotvec(np.asarray(relative_rot_vec, dtype=float).reshape(3)).as_quat()

        current_pos = self.get_eef_position()
        current_quat = self.get_eef_orientation()

        target_pos = current_pos + relative_position
        target_quat = (Rotation.from_quat(current_quat) * Rotation.from_quat(relative_quat)).as_quat()
        self.move_to_pose(target_pos, target_quat, steps=15)

    def move_to_joints(self, joints):
        joints_arr = np.asarray(joints, dtype=float).reshape(-1)
        if joints_arr.size != 7:
            raise ValueError(f"move_to_joints expects 7 values, got {joints_arr.size}")

        self._warn_once(
            "move_to_joints_cartesian_proxy",
            "[FR3_ROBOT] move_to_joints uses Cartesian proxy (FK->pose) in pose-command mode.",
        )
        T_ee, _ = self._fk_with_chain(joints_arr)
        pos = T_ee[:3, 3]
        quat = Rotation.from_matrix(T_ee[:3, :3]).as_quat()
        self.move_to_pose(pos, quat, steps=20)

    def _execute_waypoints_sync(self, waypoints):
        for pos, quat in waypoints:
            self.move_to_pose(pos, quat, steps=8)

    def execute_cartesian_waypoints(self, waypoints, asynchronous=False):
        waypoints = list(waypoints)
        if not asynchronous:
            self._execute_waypoints_sync(waypoints)
            return

        if self._async_motion_thread is not None and self._async_motion_thread.is_alive():
            raise RuntimeError("An asynchronous motion is already running.")

        self._async_motion_thread = threading.Thread(
            target=self._execute_waypoints_sync,
            args=(waypoints,),
            daemon=True,
            name="fr3-waypoints",
        )
        self._async_motion_thread.start()

    def join_motion(self):
        if self._async_motion_thread is not None and self._async_motion_thread.is_alive():
            self._async_motion_thread.join()

    # ------------------------------------------------------------------
    # Gripper compatibility methods (not supported by pose command client)
    # ------------------------------------------------------------------

    def open_gripper(self):
        self._warn_once("gripper_unsupported", "[FR3_ROBOT] Gripper control is not supported by franka_pose_cmd_client.")
        self._gripper_open_estimate = True

    def open_gripper_async(self):
        return self._executor.submit(self.open_gripper)

    def close_gripper_async(self):
        return self._executor.submit(self.close_gripper)

    def close_gripper(self):
        self._warn_once("gripper_unsupported", "[FR3_ROBOT] Gripper control is not supported by franka_pose_cmd_client.")
        self._gripper_open_estimate = False

    def get_gripper_width(self):
        return np.nan

    def is_gripper_open(self, threshold=None):
        return self._gripper_open_estimate

    # ------------------------------------------------------------------
    # Compatibility hook
    # ------------------------------------------------------------------

    def trigger_recover(self):
        self._warn_once("recover_unsupported", "[FR3_ROBOT] Recover flag is not supported by franka_pose_cmd_client.")

