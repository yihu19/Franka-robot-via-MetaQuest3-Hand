
# #### test image
# import pyrealsense2 as rs
# ctx = rs.context()
# serials = [d.get_info(rs.camera_info.serial_number) for d in ctx.query_devices()]
# if serials:
#     print("Available RealSense serial numbers:", serials)
# else:
#     print("No RealSense devices found")



##### check hdf5 file
import h5py
import numpy as np

def check_hdf5_file(filepath):
    """Check the contents of an HDF5 file"""
    try:
        with h5py.File(filepath, 'r') as f:
            print(f"=== Checking file: {filepath} ===")
            print(f"File keys: {list(f.keys())}")
            
            def print_structure(name, obj):
                print(f"{name}: {type(obj).__name__}")
                if hasattr(obj, 'shape'):
                    print(f"  Shape: {obj.shape}")
                    print(f"  Dtype: {obj.dtype}")
                if hasattr(obj, 'len'):
                    print(f"  Length: {len(obj)}")
            
            print("\n=== Dataset structure ===")
            f.visititems(print_structure)
            
            # Check specific datasets
            print("\n=== Sample data ===")
            if '/observations/qpos' in f:
                qpos = f['/observations/qpos'][:]
                print(f"Joint positions shape: {qpos.shape}")
                print(f"First joint position: {qpos[0] if len(qpos) > 0 else 'No data'}")
            
            if '/action' in f:
                actions = f['/action'][:]
                print(f"Actions shape: {actions.shape}")
                print(f"First action: {actions[0] if len(actions) > 0 else 'No data'}")
            
            if '/observations/images/wrist' in f:
                wrist_imgs = f['/observations/images/wrist']
                print(f"Wrist images shape: {wrist_imgs.shape}")
                print(f"Image dtype: {wrist_imgs.dtype}")
            
            if '/observations/images/ext1' in f:
                ext1_imgs = f['/observations/images/ext1']
                print(f"Ext1 images shape: {ext1_imgs.shape}")
                print(f"Image dtype: {ext1_imgs.dtype}")

            if '/observations/images/ext1' in f:
                world_imgs = f['/observations/images/world1']
                print(f"World images shape: {world_imgs.shape}")
                print(f"Image dtype: {world_imgs.dtype}")
                
    except Exception as e:
        print(f"Error reading file: {e}")

# Check your episode file
filepath = "/media/hca_research/SXR_TOSHIBA/Data_for_Pi0/push_t/episode_20251215_114746.hdf5"
check_hdf5_file(filepath)

# You can also check file size
import os
if os.path.exists(filepath):
    file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
    print(f"\nFile size: {file_size:.2f} MB")
else:
    print(f"File {filepath} not found")