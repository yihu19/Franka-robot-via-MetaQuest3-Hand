#!/usr/bin/env python3
# spacemouse_to_robot_converter.py - SpaceMouse 6-DOF teleoperation for Franka
#
# Subscribes to spacenav/joy (published by spacenavd + spacenav_node).
# Integrates velocity commands into a Cartesian target pose sent to the robot via UDP.
#
# Axis mapping (matches frankarobotics/franka_spacemouse reference):
#   spacenav axes[1] (push forward) -> robot +X
#   spacenav axes[0] (push right)   -> robot -Y  (negate: right = -Y in robot frame)
#   spacenav axes[2] (push up)      -> robot +Z
#   axes[3..5] roll/pitch/yaw similarly
#
# Controls:
#   Push/tilt cap    -> move robot EE (velocity integration, holds when released)
#   Button 0 (left)  -> gripper trigger
#   Button 1 (right) -> pause / resume

import socket
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Joy


class SpaceMouseToRobotConverter(Node):
    def __init__(self):
        super().__init__('spacemouse_to_robot_converter')

        # Parameters
        self.declare_parameter('robot_udp_ip',     '127.0.0.1')
        self.declare_parameter('robot_udp_port',   8888)
        self.declare_parameter('linear_scale',     0.002)   # m per control cycle (50 Hz)
        self.declare_parameter('angular_scale',    0.008)   # rad per control cycle
        self.declare_parameter('smoothing_factor', 0.5)     # 0=no smoothing, ~1=very smooth
        self.declare_parameter('control_rate',     50.0)

        self.robot_udp_ip     = self.get_parameter('robot_udp_ip').value
        self.robot_udp_port   = self.get_parameter('robot_udp_port').value
        self.linear_scale     = self.get_parameter('linear_scale').value
        self.angular_scale    = self.get_parameter('angular_scale').value
        self.smoothing_factor = self.get_parameter('smoothing_factor').value
        self.control_rate     = self.get_parameter('control_rate').value

        # Gripper
        self.gripper_state = np.array([0.0, 0.1, 20.0, 0.04, 0.04, 0.0])
        self._gripper_trigger = False

        # Latest raw axes from spacenav/joy (robot frame)
        self._raw_linear  = np.zeros(3)
        self._raw_angular = np.zeros(3)
        self._joy_received = False

        # Low-pass filtered velocity
        self._smooth_linear  = np.zeros(3)
        self._smooth_angular = np.zeros(3)

        # Accumulated target pose
        self.target_position    = np.zeros(3)
        self.target_orientation = np.array([0.0, 0.0, 0.0, 1.0])

        # Pause / resume
        self.is_paused   = False
        self._paused_pos = np.zeros(3)
        self._paused_ori = np.array([0.0, 0.0, 0.0, 1.0])

        # Button edge detection
        self._last_btn0 = False
        self._last_btn1 = False

        # Networking
        self.robot_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # ROS
        self.joy_sub    = self.create_subscription(Joy, 'spacenav/joy', self._joy_cb, 10)
        self.target_pub = self.create_publisher(PoseStamped, 'robot_target_pose', 10)
        self.timer      = self.create_timer(1.0 / self.control_rate, self._control_loop)

        # Logging
        self._joy_count  = 0
        self._last_log_t = time.time()

        self.get_logger().info('SpaceMouse to Robot Converter started')
        self.get_logger().info(f'  Robot UDP : {self.robot_udp_ip}:{self.robot_udp_port}')
        self.get_logger().info(f'  Subscribing to spacenav/joy')
        self.get_logger().info(f'  linear_scale={self.linear_scale}  angular_scale={self.angular_scale}')
        self.get_logger().info('  Left button  = gripper trigger')
        self.get_logger().info('  Right button = pause / resume')

    # ── Joy callback ─────────────────────────────────────────────────────────

    def _joy_cb(self, msg: Joy):
        axes = msg.axes
        # spacenav axis layout: [tx, ty, tz, rx, ry, rz]
        # Map to robot frame: push-forward (+Y spacenav) -> robot +X
        #                     push-right   (+X spacenav) -> robot -Y
        #                     push-up      (+Z spacenav) -> robot +Z
        if len(axes) >= 6:
            self._raw_linear  = np.array([ axes[1], -axes[0],  axes[2]])
            self._raw_angular = np.array([ axes[4], -axes[3],  axes[5]])
        self._joy_received = True
        self._joy_count += 1

        btns = msg.buttons
        # Left button -> gripper trigger (rising edge)
        btn0 = bool(btns[0]) if len(btns) > 0 else False
        if btn0 and not self._last_btn0:
            self._gripper_trigger = True
        self._last_btn0 = btn0

        # Right button -> pause/resume toggle (rising edge)
        btn1 = bool(btns[1]) if len(btns) > 1 else False
        if btn1 and not self._last_btn1:
            self.is_paused = not self.is_paused
            if self.is_paused:
                self._paused_pos = self.target_position.copy()
                self._paused_ori = self.target_orientation.copy()
                # Zero velocity buffer so robot doesn't drift on resume
                self._smooth_linear  = np.zeros(3)
                self._smooth_angular = np.zeros(3)
                self.get_logger().info('PAUSED')
            else:
                self.get_logger().info('RESUMED')
        self._last_btn1 = btn1

    # ── Control loop ─────────────────────────────────────────────────────────

    def _control_loop(self):
        if not self._joy_received:
            return

        if self.is_paused:
            self.target_position    = self._paused_pos.copy()
            self.target_orientation = self._paused_ori.copy()
        else:
            # Low-pass filter the velocity
            alpha = 1.0 - self.smoothing_factor
            self._smooth_linear  = (self.smoothing_factor * self._smooth_linear
                                    + alpha * self._raw_linear)
            self._smooth_angular = (self.smoothing_factor * self._smooth_angular
                                    + alpha * self._raw_angular)

            # Integrate position
            self.target_position += self._smooth_linear * self.linear_scale

            # Integrate orientation
            delta_rot = Rotation.from_euler('xyz', self._smooth_angular * self.angular_scale)
            self.target_orientation = (
                delta_rot * Rotation.from_quat(self.target_orientation)
            ).as_quat()
            self.target_orientation /= np.linalg.norm(self.target_orientation)

        # Gripper one-shot
        self.gripper_state[0] = 1.0 if self._gripper_trigger else 0.0
        self._gripper_trigger = False

        self._send_robot_command(self.target_position, self.target_orientation, self.gripper_state)
        self._publish_target(self.target_position, self.target_orientation)

        # Periodic log
        now = time.time()
        if now - self._last_log_t >= 2.0:
            hz = self._joy_count / (now - self._last_log_t)
            status = 'PAUSED' if self.is_paused else 'ACTIVE'
            p = self.target_position
            v = self._smooth_linear
            self.get_logger().info(
                f'spacenav {hz:.0f} Hz | {status} | '
                f'pos:[{p[0]:.3f},{p[1]:.3f},{p[2]:.3f}] '
                f'vel:[{v[0]:.3f},{v[1]:.3f},{v[2]:.3f}]'
            )
            self._joy_count  = 0
            self._last_log_t = now

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _send_robot_command(self, pos, ori, gs):
        msg = (
            f'{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} '
            f'{ori[0]:.6f} {ori[1]:.6f} {ori[2]:.6f} {ori[3]:.6f} '
            f'{gs[0]:.6f} {gs[1]:.6f} {gs[2]:.6f} {gs[3]:.6f} {gs[4]:.6f} {gs[5]:.6f}'
        )
        self.robot_socket.sendto(msg.encode(), (self.robot_udp_ip, self.robot_udp_port))

    def _publish_target(self, pos, ori):
        msg = PoseStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.x    = float(pos[0])
        msg.pose.position.y    = float(pos[1])
        msg.pose.position.z    = float(pos[2])
        msg.pose.orientation.x = float(ori[0])
        msg.pose.orientation.y = float(ori[1])
        msg.pose.orientation.z = float(ori[2])
        msg.pose.orientation.w = float(ori[3])
        self.target_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SpaceMouseToRobotConverter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
