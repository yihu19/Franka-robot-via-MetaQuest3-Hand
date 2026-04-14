#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    package_dir = get_package_share_directory('franka_vr_teleop')
    rviz_config = os.path.join(package_dir, 'config', 'pose_viz.rviz')

    return LaunchDescription([
        DeclareLaunchArgument('robot_udp_ip',     default_value='127.0.0.1'),
        DeclareLaunchArgument('robot_udp_port',   default_value='8888'),
        DeclareLaunchArgument('linear_scale',     default_value='0.001',
                              description='m per control cycle per unit axis (50Hz => 0.05m/s max)'),
        DeclareLaunchArgument('angular_scale',    default_value='0.005',
                              description='rad per control cycle per unit axis'),
        DeclareLaunchArgument('smoothing_factor', default_value='0.5',
                              description='velocity low-pass: 0=no smoothing, 1=frozen'),

        # SpaceMouse driver (spacenavd daemon must be running: sudo systemctl start spacenavd)
        Node(
            package='spacenav',
            executable='spacenav_node',
            name='spacenav',
            output='screen',
        ),

        # Converter node
        Node(
            package='franka_vr_teleop',
            executable='spacemouse_to_robot_converter',
            name='spacemouse_to_robot_converter',
            parameters=[{
                'robot_udp_ip':     LaunchConfiguration('robot_udp_ip'),
                'robot_udp_port':   LaunchConfiguration('robot_udp_port'),
                'linear_scale':     LaunchConfiguration('linear_scale'),
                'angular_scale':    LaunchConfiguration('angular_scale'),
                'smoothing_factor': LaunchConfiguration('smoothing_factor'),
                'control_rate':     50.0,
            }],
            output='screen',
        ),

        # RViz
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config],
            output='screen',
        ),
    ])
