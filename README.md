# Franka Teleoperation via Meta Quest3 Hand

The code is for franka robot teleoperation via Meta Quest3. The dataset collection is controlled via keyboard.

Please refer to https://github.com/wengmister/franka-vr-teleop and https://github.com/RollingOat/franka_research_teleoperation_with_meta_quest_vr for teleoperation with hand tracking.


# Setup Instructions

## App Download Meta Quest3

You can download the app from [Meta Quest Store](https://www.meta.com/experiences/hand-tracking-streamer/26303946202523164)



## VR Robot Client Setup

#### Build:
```bash
mkdir vr_robot_client/build && cd vr_robot_client/build
cmake -DFRANKA_INSTALL_PATH=/your/path/to/libfranka/install ..
make -j4

```

# Usage


### 1. Start VR Robot Client

```bash
cd vr_robot_client/build
./franka_vr_control_client <robot-hostname> [bidexhand]
```

When [bidexhand] is set to `true`, IK solver will limit the joint range of J7 to prevent damaging the servo sleeve attachment. Argument currently defaults to true.



### 2. Start VR Application

Put on your VR headset and start the application `Hand Tracking Streamer` that streams hand tracking data to the ROS2 node.



### 3. Launch ROS2 Node

```bash
# on ROS2 workstation
. install/setup.bash
ros2 launch franka_vr_teleop vr_control.launch.py
```

The robot teleop will be live now!


### 4. Start Collecting Dataset

```bash
cd collect_data
python run_collection.py

# Keyboard: 'R' for start recording, 'S' for Save & Stop Recording, 'Q' for Quit
```

