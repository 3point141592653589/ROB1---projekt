from pathlib import Path

import cv2 as cv
import numpy as np
from ctu_crs import CRS97

from hardware_control import capture_grid, init_robot
from utils.tray import TRAY_ID2_CENTER, get_tray_dict
from vision import ArucoPoseSolver

z_angles = []
id0 = 7
id1 = id0 + 1
tray_imgs_dir = Path(f"./datasets/deska_{id0}_{id1}")

cam_param_dir = Path("./calibration_data/cam_params/")
K = np.load(cam_param_dir / "K.npy")
dist = np.load(cam_param_dir / "dist.npy")
c2b = np.load("./calibration_data/handeye_output_refined/cam2base.npy")
dh_off = np.load("./calibration_data/handeye_output_refined/dh_offset.npy")
robot = init_robot(CRS97, home=True)
robot.dh_offset = dh_off

for imgfile in tray_imgs_dir.glob("*"):
    img = cv.imread(str(imgfile))
    t2c = ArucoPoseSolver(get_tray_dict(id0, id1), K, dist)
    transform = c2b @ t2c
    corn1 = np.zeros(3)
    corn2 = np.array([*TRAY_ID2_CENTER, 0])
    capture_grid(
        corn1,
        corn2,
        robot,
        pose=np.eye(4),
        z_axis_rotations=z_angles,
        transform=transform,
    )
