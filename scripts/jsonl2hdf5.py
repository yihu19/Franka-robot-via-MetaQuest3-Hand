import json
import h5py
import numpy as np
import os
import argparse

def convert_jsonl_to_hdf5(jsonl_path, hdf5_path):
    data_buffer = {}
    
    print(f"Reading from {jsonl_path}...")
    with open(jsonl_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            for key, value in entry.items():
                if key not in data_buffer:
                    data_buffer[key] = []
                data_buffer[key].append(value)

    print(f"Writing to {hdf5_path}...")
    with h5py.File(hdf5_path, 'w') as f:
        # Determine demo name strictly for the structure
        # user asked for data/demo_0/obs
        # We will try to infer from filename, defaulting to demo_0 if input is demo0.jsonl
        base_name = os.path.splitext(os.path.basename(jsonl_path))[0]
        if base_name == "demo0":
            demo_group_name = "demo_0"
        else:
            demo_group_name = base_name

        # Structure: data/demo_0/obs
        obs_path = f"data/{demo_group_name}/obs"
        obs_group = f.create_group(obs_path)

        for key, value_list in data_buffer.items():
            # Convert list to numpy array
            # Handle mixed types or nested lists if necessary, but assuming uniform float/list data from preview
            try:
                data_arr = np.array(value_list)
                
                if key == "robot0_gripper_qpos":
                    # Create robot0_gripper_qpos: [width/2, -width/2]
                    # The input is usually total width, so divide by 2 for finger positions
                    half_width = data_arr
                    gripper_qpos = np.stack([half_width, -half_width], axis=1)
                    obs_group.create_dataset("robot0_gripper_qpos", data=gripper_qpos)
                    print(f"  Added dataset: {obs_path}/robot0_gripper_qpos shape={gripper_qpos.shape}")
                else:
                    obs_group.create_dataset(key, data=data_arr)
                    print(f"  Added dataset: {obs_path}/{key} shape={data_arr.shape}")
            except Exception as e:
                print(f"  Failed to create dataset for key '{key}': {e}")
                
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSONL to HDF5 with specific structure.")
    parser.add_argument("input_file", nargs='?', default="low_dim_data/demo_1.jsonl", help="Input JSONL file path")
    parser.add_argument("output_file", nargs='?', default="low_dim_data/demo_1.hdf5", help="Output HDF5 file path")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        # try absolute path if relative fails, or relative to workspace root
        # heuristic: if running from scripts/ might need ../
        if os.path.exists(os.path.join("..", args.input_file)):
             args.input_file = os.path.join("..", args.input_file)
             if args.output_file == "low_dim_data/demo0.hdf5": # update default output if needed
                 args.output_file = os.path.join("..", args.output_file)

    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} not found.")
    else:
        convert_jsonl_to_hdf5(args.input_file, args.output_file)
