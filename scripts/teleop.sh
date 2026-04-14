#!/bin/bash

# Define Session Name
SESSION="franka_teleop"

# 1. Start a new tmux session and name it
tmux new-session -d -s $SESSION

# 2. Setup Pane 1: Franka Robot Client
# Navigates to the directory and starts the client
tmux send-keys -t $SESSION "cd ~/franka-vr-teleop/robot_client/build" C-m
tmux send-keys -t $SESSION "./franka_vr_control_recover_client 192.168.2.12" C-m

# 3. Wait for 10 seconds
echo "Waiting 10 seconds for the Franka client to initialize..."
sleep 10

# 4. Send "Enter" to Pane 1
# This simulates pressing the Enter key in that specific panel
tmux send-keys -t $SESSION C-m

# 5. Split the window to create Pane 2
tmux split-window -v -t $SESSION

# 6. Setup Pane 2: ROS2 Teleop Receiver
# Navigates, sources, and launches the ROS2 node
tmux send-keys -t $SESSION "cd ~/franka-vr-teleop/teleop_ws" C-m
tmux send-keys -t $SESSION "source install/setup.bash" C-m
tmux send-keys -t $SESSION "ros2 run franka_vr_teleop vr_to_robot_converter_recover --ros-args -p vr_udp_ip:=192.168.0.214 -p vr_udp_port:=9000" C-m
# 7. Attach to the session to monitor both
tmux attach-session -t $SESSION