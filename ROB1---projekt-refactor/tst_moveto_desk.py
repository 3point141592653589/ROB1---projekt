from pathlib import Path

import numpy as np
from ctu_crs import CRS97
import cv2 as cv

from hardware_control.camera import init_camera
from hardware_control.robot import init_robot
from hardware_control.robot.rotations import euler2mat
from hardware_control.robot.move import move_rotated_ik
from utils.tray import get_tray_dict
from utils.offsets import patch_target2base
from vision import ArucoPoseSolver, get_marker_pose

z_angle = 0
id0 = 7
id1 = id0 + 1
take_image = False

cam_param_dir = Path("./calibration_data/cam_params/")
K = np.load(cam_param_dir / "K.npy")
dist = np.load(cam_param_dir / "dist.npy")
c2b = np.load("./calibration_data/handeye_output_refined/cam2base.npy")
dh_off = np.load("./calibration_data/handeye_output_refined/dh_offset.npy")
robot = init_robot(CRS97, home=False)
robot.dh_offset = dh_off
# robot.soft_home()
if take_image:
    camera = init_camera()
    img = camera.grab_image()
    camera.close()
    if img is None:
        raise RuntimeError("no image was captured")
    cv.imshow("deska", img)
    cv.waitKey(0)
    cv.imwrite("deska_test.png", img)
else:
    img = cv.imread("./deska_test.png")


solver = ArucoPoseSolver(get_tray_dict(id0, id1), K, dist)
t2c = get_marker_pose(img, solver)
t2c_lift = np.eye(4)
t2c_lift[:3, 3] = np.array([0.18, 0.14, 0.120])
#t2c_lift[:3, :3] = euler2mat([(np.pi, "y")])[:3, :3]
t2b = c2b @ t2c
pose =  patch_target2base(t2b @ t2c_lift)
print(move_rotated_ik(robot, pose, [z_angle]))
