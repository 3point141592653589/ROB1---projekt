from pathlib import Path

import numpy as np
from ctu_crs import CRS97

from hardware_control.camera import init_camera
from hardware_control.robot import init_robot
from hardware_control.robot.move import move_rotated_ik
from utils.tray import get_tray_dict
from vision import ArucoPoseSolver

z_angle = 0
id0 = 7
id1 = id0 + 1

cam_param_dir = Path("./calibration_data/cam_params/")
K = np.load(cam_param_dir / "K.npy")
dist = np.load(cam_param_dir / "dist.npy")
c2b = np.load("./calibration_data/handeye_output_refined/cam2base.npy")
dh_off = np.load("./calibration_data/handeye_output_refined/dh_offset.npy")
robot = init_robot(CRS97, home=True)
robot.dh_offset = dh_off

camera = init_camera()
img = camera.grab_image()
camera.close()
if img is None:
    raise RuntimeError("no image was captured")

t2c = ArucoPoseSolver(get_tray_dict(id0, id1), K, dist)
t2c_lift = np.eye(4)
t2c_lift[2, 3] = 0.12
pose = c2b @ t2c @ t2c_lift
move_rotated_ik(robot, pose, [z_angle])
