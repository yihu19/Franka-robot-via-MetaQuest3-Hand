import os
import h5py
import argparse
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")   # avoid Wayland plugin lookup

import cv2
import time
import glob
from franka_robot import FrankaRobot


def list_episode_files(data_directory):
    """List all episode files in the data directory"""
    pattern = os.path.join(data_directory, "episode_*.hdf5")
    episode_files = glob.glob(pattern)
    episode_files.sort()  # Sort files chronologically
    return episode_files


def select_episode(episode_files):
    """Display available episodes and let user select one"""
    if not episode_files:
        print("No episode files found in the directory!")
        return None

    print("\nAvailable episodes:")
    print("-" * 50)
    for idx, file_path in enumerate(episode_files):
        filename = os.path.basename(file_path)
        # Extract timestamp from filename for better display
        timestamp = filename.replace("episode_", "").replace(".hdf5", "")
        print(f"{idx + 1:2d}. {filename} ({timestamp})")

    print("-" * 50)

    while True:
        try:
            choice = input(
                f"\nSelect episode (1-{len(episode_files)}) or 'q' to quit: ").strip()

            if choice.lower() == 'q':
                return None

            choice_idx = int(choice) - 1

            if 0 <= choice_idx < len(episode_files):
                selected_file = episode_files[choice_idx]
                print(f"\nSelected: {os.path.basename(selected_file)}")
                return selected_file
            else:
                print(
                    f"Please enter a number between 1 and {len(episode_files)}")

        except ValueError:
            print("Please enter a valid number or 'q' to quit")


def main(args):
    # Default data directory - you can modify this path
    data_directory = args.get('data_dir')

    if not os.path.isdir(data_directory):
        print(f'Data directory does not exist at \n{data_directory}\n')
        exit()

    # List and select episode
    episode_files = list_episode_files(data_directory)
    dataset_path = select_episode(episode_files)

    if dataset_path is None:
        print("No episode selected. Exiting.")
        return

    if not os.path.isfile(dataset_path):
        print(f'Selected dataset does not exist at \n{dataset_path}\n')
        exit()
    else:
        print(f'Loading dataset: \n{dataset_path}\n')

    robot = FrankaRobot()
    robot.move_home()

    with h5py.File(dataset_path, 'r') as root:
        actions = root['/action'][()]
        tm = root['/tm'][()]
        camera_images1 = root['/observations/images/wrist'][()]
        camera_images2 = root['/observations/images/ext1'][()]
        # camera_images3 = root['/observations/images/world1'][()]
        qpos = root['/observations/qpos'][()]

        n_steps = len(qpos)
        qpos_valid = np.any(np.abs(qpos) > 1e-4)
        print(f"Episode contains {n_steps} steps  (qpos valid: {qpos_valid})")
        print("Press 'q' during image display to skip to robot execution")

        # Display images
        for idx, img in enumerate(camera_images1):
            images = np.hstack((img, camera_images2[idx]))
            color_image = np.asanyarray(images)
            cv2.imshow("Camera Images", color_image)
            time.sleep(0.05)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

        if not qpos_valid:
            print("WARNING: qpos is all zeros – episode was collected without a "
                  "live state stream. Cannot replay.")
            robot.close()
            return

        # ── Diagnostic: confirm state stream is live ──────────────────────────
        live_q = robot.get_joint_positions()
        print(f"\n[DIAG] Live joint positions before replay: {np.round(live_q, 4)}")
        print(f"[DIAG] Dataset qpos[0]:                    {np.round(qpos[0], 4)}")
        print(f"[DIAG] Max diff home → qpos[-1]:           "
              f"{np.round(np.abs(qpos[-1] - qpos[0]).max(), 4)} rad "
              f"(joint {np.abs(qpos[-1] - qpos[0]).argmax()})")
        if not robot.has_valid_state():
            print("[DIAG] WARNING: no valid state from state stream!")
        print()

        # Move to the starting joint configuration of this episode before replay
        print(f"Moving to episode start position: {np.round(qpos[0], 4)}")
        robot.send_joints(qpos[0])   # blocks until robot reaches qpos[0]
        print(f"[DIAG] At start – live q: {np.round(robot.get_joint_positions(), 4)}")

        # Replay joint positions directly – each send_joints blocks until done
        print("\nStarting robot execution (target  |  live state)...")
        for idx in range(n_steps):
            ok = robot.send_joints(qpos[idx])
            live_q = robot.get_joint_positions()
            err    = np.abs(qpos[idx] - live_q).max()
            print(f"Step {idx + 1:3d}/{n_steps}  "
                  f"target={np.round(qpos[idx], 3)}  "
                  f"live={np.round(live_q, 3)}  "
                  f"max_err={err:.4f} rad"
                  + ("  [TIMEOUT]" if not ok else ""))

        final_position = robot.get_joint_positions()
        print(f"\nexpected = {np.round(qpos[-1], 4)}")
        print(f"final    = {np.round(final_position, 4)}")
        print(f"diff     = {np.round(qpos[-1] - final_position, 4)}")

        robot.move_home()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Replay robot episodes interactively')
    parser.add_argument('--data_dir', action='store', type=str,
                        default="./data/test",
                        help='Directory containing episode files')


    main(vars(parser.parse_args()))
