import cv2
import numpy as np
import time
import pyrealsense2 as rs

# =============================================================================
# CAMERA CONFIGURATION - MODIFY THIS SECTION FOR YOUR SETUP
# =============================================================================

CAMERA_CONFIG = {
    "ext1": {
        "type": "realsense",
        "serial_number": "234322305598",
        "fps": 30,
        "width": 640,
        "height": 480
    },
    "wrist": {
        "type": "realsense",  # 从 luxonis 改成 realsense
        "serial_number": "241122306284",
        "fps": 30,
        "width": 640,
        "height": 480
    },
    # "ext2": {
    #     "type": "realsense",  # 从 luxonis 改成 realsense
    #     "serial_number": "241122302482",
    #     "fps": 30,
    #     "width": 640,
    #     "height": 480
    # },

}


# =============================================================================
# CAMERA CLASS
# =============================================================================
class Cameras:
    def __init__(self, camera_config=None):
        if camera_config is None:
            camera_config = CAMERA_CONFIG

        self.cameras = {}
        self.camera_config = camera_config
        for name, config in camera_config.items():
            if config["type"] == "realsense":
                self.cameras[name] = self._init_realsense(config)
            else:
                raise ValueError(
                    f"Unknown camera type: {config['type']} for camera {name}")

    def _init_realsense(self, config):
        ctx = rs.context()
        devices = ctx.query_devices()
        available_serials = [device.get_info(
            rs.camera_info.serial_number) for device in devices]

        if config["serial_number"] not in available_serials:
            raise Exception(
                f"RealSense camera {config['serial_number']} not found")

        pipeline = rs.pipeline()
        rs_config = rs.config()
        rs_config.enable_device(config["serial_number"])
        rs_config.enable_stream(
            rs.stream.color,
            config["width"],
            config["height"],
            rs.format.bgr8,
            config["fps"]
        )
        profile = pipeline.start(rs_config)
        return {"type": "realsense", "pipeline": pipeline, "profile": profile, "config": config}


    def get_frames(self):
        frames = {}
        for name, camera in self.cameras.items():
            if camera["type"] == "realsense":
                pipeline_frames = camera["pipeline"].wait_for_frames()
                color_frame = pipeline_frames.get_color_frame()
                if not color_frame:
                    raise Exception(f"No color frame from camera {name}")
                color_image = np.asanyarray(color_frame.get_data())
                frames[name] = cv2.resize(
                    color_image, (camera["config"]["width"], camera["config"]["height"]))

        return frames

    def get_depth_frames(self):
        depth_frames = {}
        for name, camera in self.cameras.items():
            if camera["type"] == "realsense":
                pipeline_frames = camera["pipeline"].wait_for_frames()
                depth_frame = pipeline_frames.get_depth_frame()
                if not depth_frame:
                    raise Exception(f"No depth frame from camera {name}")
                depth_image = np.asanyarray(depth_frame.get_data())
                depth_frames[name] = cv2.resize(
                    depth_image, (camera["config"]["width"], camera["config"]["height"]))
        return depth_frames

    def get_intrinsics(self):
        intrinsics = {}
        for name, camera in self.cameras.items():
            if camera["type"] == "realsense":
                profile = camera["profile"]
                color_stream = profile.get_stream(rs.stream.color)
                color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
                depth_stream = profile.get_stream(rs.stream.depth)
                depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
                intrinsics[name] = {
                    "color": {"fx": color_intrinsics.fx, "fy": color_intrinsics.fy, "cx": color_intrinsics.ppx, "cy": color_intrinsics.ppy, "distortion": color_intrinsics.coeffs},
                    "depth": {"fx": depth_intrinsics.fx, "fy": depth_intrinsics.fy, "cx": depth_intrinsics.ppx, "cy": depth_intrinsics.ppy, "distortion": depth_intrinsics.coeffs}
                }
        return intrinsics

    def get_extrinsics(self):
        extrinsics = {}
        for name in self.cameras.keys():
            extrinsics[name] = None
        return extrinsics

    def close(self):
        for camera in self.cameras.values():
            if camera["type"] == "realsense":
                camera["pipeline"].stop()

    def __del__(self):
        self.close()
