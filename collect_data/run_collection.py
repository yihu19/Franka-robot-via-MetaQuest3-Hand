"""
run_collection.py – Data collection main loop.

The robot is teleoperated externally (VR / Meta Quest pipeline).
This script only reads robot state via the UDP state stream and records
episodes to HDF5.  Keyboard keys control recording:

    r  – start a new recording episode
    s  – save & stop the current episode
    q  – quit

The recorded "action" field is the EEF pose delta (velocity) computed from
consecutive state readings: [vx, vy, vz, wx, wy, wz, gripper].
Gripper is set to 0.0 because gripper state is not in the UDP state stream.
"""

import os
import signal
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")   # avoid Wayland plugin lookup

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from franka_robot import FrankaRobot, RobotInputs
from game_controller import KeyboardController
from cameras import Cameras
from data_recorder import DataRecorder

DEFAULT_STATE_PORT = int(
    os.environ.get("FRANKA_STATE_PORT", str(FrankaRobot.STATE_PORT))
)
DEBUG_STATE = os.environ.get("COLLECT_DATA_DEBUG_STATE", "0") == "1"


# ---------------------------------------------------------------------------
# Action computation
# ---------------------------------------------------------------------------

def compute_action(curr_ee_t, curr_ee_q, prev_ee_t, prev_ee_q, dt):
    """
    Compute action vector from consecutive EEF states.

    Returns RobotInputs with [vx, vy, vz, wx, wy, wz, gripper=0].
    On the first call (prev_* is None) returns a zero RobotInputs.
    """
    if prev_ee_t is None or dt <= 0.0:
        return RobotInputs()

    delta_t = (curr_ee_t - prev_ee_t) / dt

    curr_rot = Rotation.from_quat(curr_ee_q)
    prev_rot = Rotation.from_quat(prev_ee_q)
    ang_vel = (curr_rot * prev_rot.inv()).as_rotvec() / dt

    return RobotInputs(
        [delta_t[0], delta_t[1], delta_t[2],
         ang_vel[0], ang_vel[1], ang_vel[2],
         0.0],          # gripper not available via UDP state stream
        False, False,
    )


# ---------------------------------------------------------------------------
# Signal handler
# ---------------------------------------------------------------------------

def signal_handler(sig, frame):
    print("\nCtrl+C detected. Exiting safely...")
    cv2.destroyAllWindows()
    sys.exit(0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    keyboard = KeyboardController()
    robot = FrankaRobot(state_port=DEFAULT_STATE_PORT)
    cameras = Cameras()
    recorder = DataRecorder(robot, cameras)

    is_recording = False
    prev_ee_t = None
    prev_ee_q = None
    previous = time.time()
    waiting_for_valid_state = False
    state_wait_last_log = 0.0

    print(f"\nRobot state reader connected (UDP port {DEFAULT_STATE_PORT})")
    print("Camera feed will be displayed in separate windows")
    print("Beginning collection loop\n")

    try:
        while True:
            t0 = time.time()

            start_rec, stop_rec, do_quit = keyboard.get_recording_controls()

            if do_quit:
                print("Quit key pressed. Exiting.")
                break

            if start_rec and not is_recording:
                recorder.reset()
                is_recording = True
                prev_ee_t = None
                prev_ee_q = None
                previous = time.time()
                waiting_for_valid_state = True
                state_wait_last_log = 0.0
                print("Recording started")

            elif stop_rec and is_recording:
                print("Saving recording...")
                try:
                    recorder.save_data()
                except Exception as e:
                    print(f"Error saving recording: {e}")
                is_recording = False
                print("Recording stopped")

            if is_recording:
                action_tm = time.time() - previous
                previous = time.time()

                state_diag = robot.get_state_diagnostics()
                if not robot.has_valid_state():
                    now = time.time()
                    if DEBUG_STATE or (now - state_wait_last_log) >= 1.0:
                        print(
                            f"[STATE_WAIT] waiting for valid robot state on UDP {state_diag['state_port']} "
                            f"(rx_count={state_diag['rx_count']}, missing={state_diag['missing_required_keys']})"
                        )
                        state_wait_last_log = now
                    continue

                if waiting_for_valid_state:
                    print(
                        f"[STATE_WAIT] first valid state received on UDP {state_diag['state_port']} "
                        f"(rx_count={state_diag['rx_count']})"
                    )
                    waiting_for_valid_state = False

                curr_ee_t = robot.get_ee_translation()
                curr_ee_q = robot.get_ee_quaternion()

                robot_inputs = compute_action(
                    curr_ee_t, curr_ee_q, prev_ee_t, prev_ee_q, action_tm
                )
                prev_ee_t = curr_ee_t
                prev_ee_q = curr_ee_q

                recorder.record_sample(robot_inputs, action_tm)

            # Display camera frames
            try:
                camera_images = cameras.get_frames()
                if camera_images:
                    for cam_name, image in camera_images.items():
                        if image is not None:
                            cv2.imshow(f"Camera {cam_name}", image)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        print("'q' key pressed in camera window. Exiting.")
                        break
            except Exception as e:
                print(f"Error displaying camera images: {e}")

            # Maintain ~10 Hz loop rate
            dt = time.time() - t0
            time.sleep(max(0.0, 1.0 / 10.0 - dt))

    except KeyboardInterrupt:
        print("\nKeyboard interrupt. Exiting.")
    except Exception as e:
        print(f"\nUnhandled error: {e}")
    finally:
        cv2.destroyAllWindows()
        robot.close()
