import h5py
import os
from datetime import datetime
from collections import defaultdict
import numpy as np
import time

from franka_robot import FrankaRobot, RobotInputs
from cameras import Cameras


class DataRecorder:
    def __init__(self, robot: FrankaRobot, cameras: Cameras):
        self.robot = robot
        self.cameras = cameras
        self.data_dict = defaultdict(list)
        self.dataset_dir = "./data/test"
        self.camera_names = ["ext1", "wrist"]
        self.start_time = None
        self.debug_state = os.environ.get("COLLECT_DATA_DEBUG_STATE", "0") == "1"
        self.debug_stride = max(1, int(os.environ.get("COLLECT_DATA_DEBUG_EVERY", "25")))

    def reset(self):
        self.data_dict = defaultdict(list)
        self.start_time = time.time()

    def record_sample(self, robot_inputs: RobotInputs, dt):
        action = [
            robot_inputs.left_x, robot_inputs.left_y, robot_inputs.right_z,
            robot_inputs.roll, robot_inputs.pitch, robot_inputs.yaw,
            robot_inputs.gripper,
        ]

        qpos = self.robot.get_joint_positions()
        qvel = self.robot.get_joint_velocities()
        gpos = np.array([self.robot.get_gripper_width()])
        rotq = self.robot.get_ee_quaternion()
        ee_rpy = self.robot.quaternion_array_to_rpy(rotq)
        ee_t = self.robot.get_ee_translation()
        ee_twist_ang = self.robot.get_ee_twist_angular()
        ee_twist_lin = self.robot.get_ee_twist_linear()

        self.data_dict["/observations/qpos"].append(qpos)
        self.data_dict["/observations/qvel"].append(qvel)
        self.data_dict["/observations/gpos"].append(gpos)

        self.data_dict["/observations/ee_pos_q"].append(rotq)
        self.data_dict["/observations/ee_pos_rpy"].append(ee_rpy)
        self.data_dict["/observations/ee_pos_t"].append(ee_t)
        self.data_dict["/observations/ee_twist_ang"].append(ee_twist_ang)
        self.data_dict["/observations/ee_twist_lin"].append(ee_twist_lin)

        self.data_dict["/observations/elbow_jnt3_pos"].append(
            np.array([self.robot.get_elbow_joint3_pos()])
        )
        self.data_dict["/observations/elbow_jnt4_flip"].append(
            np.array([self.robot.get_elbow_joint4_flip()])
        )

        self.data_dict["/observations/O_F_ext_hat_K"].append(
            self.robot.get_O_F_ext_hat_K()
        )
        self.data_dict["/observations/tau_J"].append(self.robot.get_tau_J())
        self.data_dict["/observations/dtau_J"].append(self.robot.get_dtau_J())
        self.data_dict["/observations/tau_ext_hat_filtered"].append(
            self.robot.get_tau_ext_hat_filtered()
        )

        self.data_dict["/action"].append(action)
        self.data_dict["/tm"].append(np.array([dt]))

        sample_idx = len(self.data_dict["/observations/qpos"]) - 1
        if self.debug_state and sample_idx % self.debug_stride == 0:
            print(
                f"[DEBUG_STATE] sample {sample_idx}: "
                f"qpos={np.array2string(qpos, precision=4)} "
                f"qvel={np.array2string(qvel, precision=4)} "
                f"ee_t={np.array2string(ee_t, precision=4)}"
            )

        frames = self.cameras.get_frames()
        if self.camera_names == ["ext1", "wrist"]:
            self.data_dict["/observations/images/wrist"].append(frames["wrist"])
            self.data_dict["/observations/images/ext1"].append(frames["ext1"])

    def save_data(self):
        t0 = time.time()

        datetimes = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_path = os.path.join(self.dataset_dir, f"episode_{datetimes}")

        num_timesteps = len(self.data_dict["/observations/qpos"])
        print(f"{num_timesteps=}")

        if num_timesteps == 0:
            print("No data to save. Recording was empty.")
            return

        os.makedirs(self.dataset_dir, exist_ok=True)

        if self.debug_state:
            qpos_arr = np.asarray(self.data_dict["/observations/qpos"])
            qvel_arr = np.asarray(self.data_dict["/observations/qvel"])
            print(
                f"[DEBUG_STATE] pre-save qpos nonzero={np.count_nonzero(qpos_arr)}/{qpos_arr.size} "
                f"qvel nonzero={np.count_nonzero(qvel_arr)}/{qvel_arr.size}"
            )
            preview_count = min(3, num_timesteps)
            print(
                f"[DEBUG_STATE] pre-save qpos first {preview_count}: "
                f"{np.array2string(qpos_arr[:preview_count], precision=4)}"
            )

        with h5py.File(dataset_path + ".hdf5", "w") as root:
            obs = root.create_group("observations")
            image = obs.create_group("images")
            depth = obs.create_group("depth")

            for cam_name in self.camera_names:
                width = self.cameras.camera_config[cam_name]["width"]
                height = self.cameras.camera_config[cam_name]["height"]
                chunk_size = (1, height, width, 3) if num_timesteps > 0 else None
                image.create_dataset(
                    cam_name,
                    (num_timesteps, height, width, 3),
                    dtype="uint8",
                    chunks=chunk_size,
                )

                if f"/observations/depth/{cam_name}" in self.data_dict:
                    chunk_size = (1, height, width) if num_timesteps > 0 else None
                    depth.create_dataset(
                        cam_name,
                        (num_timesteps, height, width),
                        dtype="uint16",
                        chunks=chunk_size,
                    )

            obs.create_dataset("qpos", (num_timesteps, 7))
            obs.create_dataset("qvel", (num_timesteps, 7))
            obs.create_dataset("gpos", (num_timesteps, 1))

            obs.create_dataset("ee_pos_q", (num_timesteps, 4))
            obs.create_dataset("ee_pos_rpy", (num_timesteps, 3))
            obs.create_dataset("ee_pos_t", (num_timesteps, 3))

            obs.create_dataset("ee_twist_ang", (num_timesteps, 3))
            obs.create_dataset("ee_twist_lin", (num_timesteps, 3))

            obs.create_dataset("elbow_jnt3_pos", (num_timesteps, 1))
            obs.create_dataset("elbow_jnt4_flip", (num_timesteps, 1))

            obs.create_dataset("O_F_ext_hat_K", (num_timesteps, 6))
            obs.create_dataset("tau_J", (num_timesteps, 7))
            obs.create_dataset("dtau_J", (num_timesteps, 7))
            obs.create_dataset("tau_ext_hat_filtered", (num_timesteps, 7))

            root.create_dataset("action", (num_timesteps, 7))
            root.create_dataset("tm", (num_timesteps, 1))

            for name, array in self.data_dict.items():
                print(f"Saving: {name=}")
                root[name][...] = array

        print(f"Saving: {time.time() - t0:.1f} secs\n")
