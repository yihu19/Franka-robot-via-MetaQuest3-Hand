#!/usr/bin/env python3
# vr_to_robot_converter.py - Convert VR wrist tracking to robot pose commands

import rclpy
from rclpy.node import Node
import socket
import threading
import time
import sys
import re
import numpy as np
import struct
from scipy.spatial.transform import Rotation, Slerp
from geometry_msgs.msg import PoseStamped

class VRToRobotConverter(Node):
    def __init__(self):
        super().__init__('vr_to_robot_converter')
        
        # Parameters
        self.declare_parameter('vr_udp_ip', '0.0.0.0')
        self.declare_parameter('vr_udp_port', 9000)
        
        self.declare_parameter('robot_udp_ip', '192.168.18.1')
        self.declare_parameter('robot_udp_port', 8888)
        self.declare_parameter('smoothing_factor', 0.7)
        self.declare_parameter('control_rate', 100.0)  # Hz
        
        # Get parameters
        self.vr_udp_ip = self.get_parameter('vr_udp_ip').value
        self.vr_udp_port = self.get_parameter('vr_udp_port').value
        self.robot_udp_ip = self.get_parameter('robot_udp_ip').value
        self.robot_udp_port = self.get_parameter('robot_udp_port').value
        self.smoothing_factor = self.get_parameter('smoothing_factor').value
        self.control_rate = self.get_parameter('control_rate').value
        
        # Unity LH (x right, y up, z forward) -> RH (x front, y left, z up)
        self._unity_to_rh = np.array([
            [0.0,  0.0,  1.0],
            [-1.0, 0.0,  0.0],
            [0.0,  1.0,  0.0],
        ])

        # Pinch threshold for gripper close (meters)
        self.pinch_close_dist = 0.03

        # gripper state
        self.index_button_pressed = 0.0
        self.gripper_speed  = 0.1  # m/s
        self.gripper_force  = 20.0   # N
        self.epsilon_inner = 0.04  # m
        self.epsilon_outer = 0.04  # m
        self.gripper_grasp_width = 0.0 # m
        self.gripper_state  = np.array([self.index_button_pressed, self.gripper_speed, self.gripper_force, self.epsilon_inner, self.epsilon_outer, self.gripper_grasp_width]) # position m, speed m/s, force N

        # VR state
        self.current_vr_pose = None
        self.initial_vr_pose = None
        self.vr_data_received = False

        # Target robot pose
        self.target_position = np.array([0.0, 0.0, 0.0])
        self.target_orientation = np.array([0.0, 0.0, 0.0, 1.0])
        
        # Robot state - absolute pose (not delta)
        self.robot_base_pose = {
            'position': np.array([0.0, 0.0, 0.0]),
            'orientation': np.array([0.0, 0.0, 0.0, 1.0])  # identity quaternion
        }
        
        # Smoothing
        self.smoothed_position = np.array([0.0, 0.0, 0.0])
        self.smoothed_orientation = np.array([0.0, 0.0, 0.0, 1.0])
        
        # Pause state control
        self.pause_button_pressed = 0.0
        self.is_paused = False
        self.paused_vr_pose = {
                    'position': np.array([0.0, 0.0, 0.0]),
                    'orientation': np.array([0.0, 0.0, 0.0, 1.0])
                }
        
        # State machine flags
        self.control_started = False
        self.last_button_state = False
        
        # Setup VR UDP receiver
        try:
            self.vr_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.vr_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.vr_socket.bind((self.vr_udp_ip, self.vr_udp_port))
            self.vr_socket.setblocking(False)
            self.get_logger().info(f"Successfully bound VR UDP socket to {self.vr_udp_ip}:{self.vr_udp_port}")
        except Exception as e:
            self.get_logger().error(f"Failed to setup VR UDP socket: {str(e)}")
            raise
        
        # Setup robot UDP sender
        self.robot_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # Start VR receiver thread
        self.vr_thread = threading.Thread(target=self.receive_vr_data)
        self.vr_thread.daemon = True
        self.vr_thread.start()
        
        # Control loop timer
        self.timer = self.create_timer(1.0/self.control_rate, self.control_loop)
        
        # Publishers for visualization
        self.vr_pose_pub = self.create_publisher(PoseStamped, 'vr_wrist_pose', 10)
        self.robot_target_pub = self.create_publisher(PoseStamped, 'robot_target_pose', 10)
        
        # Frequency tracking
        self.command_counter = 0
        self.last_log_time = time.time()
        self.commands_sent = 0
        self.vr_messages_received = 0  # Track VR message frequency
        self.log_interval = 2.0  # Log every 2 seconds
        
        self.get_logger().info(f"VR to Robot Converter started")
        self.get_logger().info(f"VR UDP: {self.vr_udp_ip}:{self.vr_udp_port}")
        self.get_logger().info(f"Robot UDP: {self.robot_udp_ip}:{self.robot_udp_port}")
        self.get_logger().info("Move your VR hand to start control!")

    def __del__(self):
        """Cleanup resources when node is destroyed"""
        pass
    
    def receive_vr_data(self):
        """Thread function to receive VR wrist tracking data"""
        self.get_logger().info('VR UDP receiver thread started')
        last_debug_time = time.time()
        message_count = 0
        
        while rclpy.ok():
            try:
                data, addr = self.vr_socket.recvfrom(1024)
                message_count += 1

                # If we received exactly 80 bytes, treat it as binary pose (10 doubles)
                if len(data) == 80:
                    self.parse_vr_binary(data, addr)
                else:
                    # Try to parse as UTF-8 text from HandLandmarkStreamer.cs
                    try:
                        self.parse_vr_text(data.decode('utf-8'))
                    except Exception as e:
                        print(f"WARNING: received unrecognised data (len={len(data)}): {e}")
                
                # Debug logging every 5 seconds
                current_time = time.time()
                if current_time - last_debug_time >= 5.0:
                    self.get_logger().info(f"Received {message_count} VR messages in last 5 seconds")
                    last_debug_time = current_time
                    message_count = 0
                
            except BlockingIOError:
                # No data available, check for debug timeout
                current_time = time.time()
                if current_time - last_debug_time >= 5.0:
                    if message_count == 0:  # Only log if truly no messages received
                        self.get_logger().info("No VR data received in last 5 seconds")
                    last_debug_time = current_time
                    message_count = 0
                time.sleep(0.001)
            except Exception as e:
                self.get_logger().error(f'Error receiving VR data: {str(e)}')

                    
    def parse_vr_binary(self, data, addr=None):
        """Parse 80-byte binary pose: 10 doubles
        (x,y,z, qx,qy,qz,qw, press_index, pause_val, extra)
        Assume binary pose is also in Unity LH coordinates.
        """
        try:
            if len(data) != 80:
                self.get_logger().warn(f"Binary pose of unexpected length {len(data)}")
                return

            # Unpack 10 little-endian doubles
            vals = struct.unpack('<10d', data)
            x, y, z, qx, qy, qz, qw, press_index, pause_val, _ = vals

            # Increment VR message counter
            self.vr_messages_received += 1

            # ---------------------------------------------------------
            # Convert Unity LH (x right, y up, z forward)
            # to robot RH (x front, y left, z up)
            # ---------------------------------------------------------
            unity_position = np.array([x, y, z])
            robot_position = self._unity_to_rh @ unity_position

            vr_rot = Rotation.from_quat([qx, qy, qz, qw])
            robot_rot = Rotation.from_matrix(
                self._unity_to_rh @ vr_rot.as_matrix() @ self._unity_to_rh.T
            )
            robot_quat = robot_rot.as_quat()

            self.current_vr_pose = {
                'position': robot_position,
                'orientation': robot_quat
            }

            # Gripper button
            self.index_button_pressed = 1.0 if press_index > 0.5 else 0.0
            self.gripper_state[0] = float(self.index_button_pressed)

            # Pause button
            self.pause_button_pressed = 1.0 if pause_val > 0.5 else 0.0

            if not self.vr_data_received:
                self.initial_vr_pose = self.current_vr_pose.copy()
                self.vr_data_received = True
                self.get_logger().info("Initial VR pose captured (binary, Unity->RH)!")

            # Publish VR pose for visualization
            self.publish_vr_pose(robot_position, robot_quat)

        except Exception as e:
            self.get_logger().error(f'Error parsing binary VR pose: {str(e)}')



    def parse_vr_text(self, text: str):
        """Parse UTF-8 text from HandLandmarkStreamer.cs (wrist + landmarks) and
        FistTracking.cs (fist state for pause control).

        Wrist line:     'Right wrist:, x, y, z, qx, qy, qz, qw'
        Landmarks line: 'Right landmarks:, x,y,z, ...' (21 joints from wrist)
        Fist line:      'Fist: closed' / 'Fist: open'
        """
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue

            parts = [p.strip() for p in line.split(',')]
            label = parts[0].lower()

            # Fist state from FistTracking.cs -> pause button
            if 'fist' in label:
                self.pause_button_pressed = 1.0 if 'closed' in label else 0.0
                continue

            # Only process right hand
            if 'right' not in label:
                continue

            # Parse float values from remaining comma-separated fields
            floats = []
            for p in parts[1:]:
                try:
                    floats.append(float(p))
                except (ValueError, AttributeError):
                    pass

            if 'wrist' in label and len(floats) >= 7:
                x, y, z = floats[0], floats[1], floats[2]
                qx, qy, qz, qw = floats[3], floats[4], floats[5], floats[6]

                # Transform Unity LH -> RH (Franka base convention)
                robot_position = self._unity_to_rh @ np.array([x, y, z])
                vr_rot = Rotation.from_quat([qx, qy, qz, qw])
                robot_rot = Rotation.from_matrix(
                    self._unity_to_rh @ vr_rot.as_matrix() @ self._unity_to_rh.T
                )
                robot_quat = robot_rot.as_quat()

                self.vr_messages_received += 1
                self.current_vr_pose = {
                    'position': robot_position,
                    'orientation': robot_quat,
                }

                if not self.vr_data_received:
                    self.initial_vr_pose = self.current_vr_pose.copy()
                    self.vr_data_received = True
                    self.get_logger().info("Initial VR pose captured (text format)!")

                self.publish_vr_pose(robot_position, robot_quat)

            elif 'landmarks' in label:
                n = len(floats) // 3
                if n < 9:
                    continue
                landmarks = np.array(floats[:n * 3]).reshape(n, 3)
                # Streamed joint order: wrist(0), thumb(1-4), index(5-8), ...
                # Thumb tip = index 4, Index tip = index 8
                pinch_dist = float(np.linalg.norm(landmarks[4] - landmarks[8]))
                self.index_button_pressed = 1.0 if pinch_dist < self.pinch_close_dist else 0.0
                self.gripper_state[0] = float(self.index_button_pressed)

    def control_loop(self):
        """Main control loop - converts VR pose to robot commands"""
        if not self.vr_data_received or self.current_vr_pose is None:
            return
        
        self.command_counter += 1
        
        try:
            # Detect button rising edge (toggle logic)
            # Use a threshold slightly higher than 0.5 to be safe
            is_button_down = self.pause_button_pressed > 0.5
            button_triggered = is_button_down and not self.last_button_state
            self.last_button_state = is_button_down

            # STATE: WAITING TO START — auto-start on first VR data
            if not self.control_started:
                self.control_started = True
                self.is_paused = False

                # Capture starting VR pose as the zero reference
                if self.current_vr_pose:
                    self.initial_vr_pose = self.current_vr_pose.copy()

                # Reset any accumulated offset to zero (starting fresh)
                self.paused_vr_pose['position'] = np.array([0.0, 0.0, 0.0])
                self.paused_vr_pose['orientation'] = np.array([0.0, 0.0, 0.0, 1.0])

                self.get_logger().info("Control AUTO-STARTED. Close left fist to pause/resume.")

            # STATE: RUNNING (ACTIVE or PAUSED)
            else:
                 # Handle toggle (Pause <-> Active)
                if button_triggered:
                    self.is_paused = not self.is_paused
                    
                    if self.is_paused:
                        # Transition to PAUSED
                        # Freeze the current target as the holding pose
                        self.paused_vr_pose['position'] = self.target_position.copy()
                        self.paused_vr_pose['orientation'] = self.target_orientation.copy()
                        self.get_logger().info("PAUSED mode. Press button again to resume.")
                        
                    else:
                        # Transition to ACTIVE (Resumed)
                        # Re-clutch: Set new initial pose to current VR pose
                        # (Motion will continue relative to this new reference, added to the last frozen pose)
                        if self.current_vr_pose:
                            self.initial_vr_pose = self.current_vr_pose.copy()
                        self.smoothed_position = self.paused_vr_pose['position'].copy()
                        self.smoothed_orientation = self.paused_vr_pose['orientation'].copy()
                        self.get_logger().info("RESUMED control.")

            # Calculate Target Pose
            # If paused, continue using the paused position/orientation
            if self.is_paused:
                self.target_position = self.paused_vr_pose['position']
                self.target_orientation = self.paused_vr_pose['orientation']
                # Let the robot’s internal controller hold its position naturally.
                # return 
            else:
                # Active calculation
                # Calculate pose difference from reference VR pose (apply paused offset)
                vr_pos_delta = self.current_vr_pose['position'] - self.initial_vr_pose['position'] + self.paused_vr_pose['position']

                # Apply smoothing to position
                self.smoothed_position = (self.smoothing_factor * self.smoothed_position + 
                                        (1 - self.smoothing_factor) * vr_pos_delta)

                # Calculate orientation difference as relative rotation
                initial_rot = Rotation.from_quat(self.initial_vr_pose['orientation'])
                current_rot = Rotation.from_quat(self.current_vr_pose['orientation'])
                relative_rot = current_rot * initial_rot.inv()

                # Apply relative rotation from paused VR pose
                paused_rot = Rotation.from_quat(self.paused_vr_pose['orientation'])
                target_rot = paused_rot * relative_rot

                # Slerp between current smoothed orientation and target orientation
                slerp_t = 1 - self.smoothing_factor
                current_smoothed_rot = Rotation.from_quat(self.smoothed_orientation)
                key_rotations = Rotation.from_quat([current_smoothed_rot.as_quat(), target_rot.as_quat()])
                slerp = Slerp([0, 1], key_rotations)
                smoothed_rot = slerp(slerp_t)
                self.smoothed_orientation = smoothed_rot.as_quat()
                # Normalize quaternion
                self.smoothed_orientation = self.smoothed_orientation / np.linalg.norm(self.smoothed_orientation)

                # Calculate absolute target pose (base + delta)
                self.target_position = self.robot_base_pose['position'] + self.smoothed_position
                self.target_orientation = self.smoothed_orientation
            
            # Ensure gripper state first element is up-to-date (float 0.0 or 1.0)
            self.gripper_state[0] = float(self.index_button_pressed)

            # Send robot command
            self.send_robot_command(self.target_position, self.target_orientation, self.gripper_state)

            # after sending the command, reset the index button pressed flag
            self.index_button_pressed = 0.0
            self.gripper_state[0] = 0.0
            
            # Publish target pose for visualization
            self.publish_robot_target(self.target_position, self.target_orientation)
            
            # Frequency logging
            current_time = time.time()
            if current_time - self.last_log_time >= self.log_interval:
                vr_frequency = self.vr_messages_received / (current_time - self.last_log_time)
                
                status_str = "WAITING"
                if self.control_started:
                    status_str = "PAUSED" if self.is_paused else "ACTIVE"
                    
                pause_info = f'Button: {self.pause_button_pressed:.0f} ({status_str})'
                
                self.get_logger().info(
                    f'VR UDP sampling: {vr_frequency:.1f} Hz | {pause_info} | '
                    f'Target pos: [{self.target_position[0]:.3f}, {self.target_position[1]:.3f}, {self.target_position[2]:.3f}] | '
                    f'VR delta: [{self.smoothed_position[0]:.3f}, {self.smoothed_position[1]:.3f}, {self.smoothed_position[2]:.3f}]'
                )
                self.last_log_time = current_time
                self.commands_sent = 0
                self.vr_messages_received = 0
            
            self.commands_sent += 1
                
        except Exception as e:
            self.get_logger().error(f'Error in control loop: {str(e)}')
    
    def send_robot_command(self, position, orientation, gripper_state):
        """Send absolute pose command to robot via UDP"""
        try:
            # Send absolute pose: position (x,y,z) and orientation (qx,qy,qz,qw) and gripper state (button_pressed, speed, force, epsilon_inner, epsilon_outer, width)
            message = f"{position[0]:.6f} {position[1]:.6f} {position[2]:.6f} " + \
                        f"{orientation[0]:.6f} {orientation[1]:.6f} {orientation[2]:.6f} {orientation[3]:.6f} " + \
                        f"{gripper_state[0]:.6f} {gripper_state[1]:.6f} {gripper_state[2]:.6f} {gripper_state[3]:.6f} {gripper_state[4]:.6f} {gripper_state[5]:.6f}"

            # Send to robot
            self.robot_socket.sendto(message.encode(), (self.robot_udp_ip, self.robot_udp_port))
            
        except Exception as e:
            self.get_logger().error(f'Error sending robot command: {str(e)}')
    
    def publish_vr_pose(self, position, orientation):
        """Publish VR pose for visualization"""
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "map"  # Use map as parent frame
        
        pose_msg.pose.position.x = float(position[0])
        pose_msg.pose.position.y = float(position[1])
        pose_msg.pose.position.z = float(position[2])
        
        pose_msg.pose.orientation.x = float(orientation[0])
        pose_msg.pose.orientation.y = float(orientation[1])
        pose_msg.pose.orientation.z = float(orientation[2])
        pose_msg.pose.orientation.w = float(orientation[3])
        
        self.vr_pose_pub.publish(pose_msg)
    
    def publish_robot_target(self, position, orientation):
        """Publish robot target pose for visualization"""
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "map"  # Use map as parent frame
        
        pose_msg.pose.position.x = float(position[0])
        pose_msg.pose.position.y = float(position[1])
        pose_msg.pose.position.z = float(position[2])
        
        pose_msg.pose.orientation.x = float(orientation[0])
        pose_msg.pose.orientation.y = float(orientation[1])
        pose_msg.pose.orientation.z = float(orientation[2])
        pose_msg.pose.orientation.w = float(orientation[3])
        
        self.robot_target_pub.publish(pose_msg)

    

def main(args=None):
    rclpy.init(args=args)
    
    node = VRToRobotConverter()
    print("VR to Robot Converter Node started.")
    # print out all the parameters
    print("VR to Robot Converter Parameters:")
    print(f"  vr_udp_ip: {node.vr_udp_ip}")
    print(f"  vr_udp_port: {node.vr_udp_port}")
    print(f"  robot_udp_ip: {node.robot_udp_ip}")
    print(f"  robot_udp_port: {node.robot_udp_port}")
    print(f"  smoothing_factor: {node.smoothing_factor}")   
    print(f"  control_rate: {node.control_rate}")
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()