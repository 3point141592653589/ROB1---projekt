from pathlib import Path

import cv2 as cv
import numpy as np
from ctu_crs import CRS97

from hardware_control import capture_grid, init_robot
from utils.tray import TRAY_ID2_CENTER, get_tray_dict
from utils.offsets import patch_target2base
from vision import ArucoPoseSolver, get_marker_pose

z_angles = [np.pi/2]
id0 = 7
id1 = id0 + 1
tray_imgs_dir = Path(f"./datasets/deska_{id0}_{id1}")

cam_param_dir = Path("./cam_params2/")
K = np.load(cam_param_dir / "K.npy")
dist = np.load(cam_param_dir / "dist.npy")
c2b = np.load("./calibration_data/handeye_output_streda_refined/cam2base.npy")
dh_off = np.load("./calibration_data/handeye_output_streda_refined/dh_offset.npy")
robot = init_robot(CRS97, home=False)
robot.dh_offset = dh_off

for imgfile in tray_imgs_dir.glob("*"):
    img = cv.imread(str(imgfile))
    solver = ArucoPoseSolver(get_tray_dict(id0, id1), K, dist)
    t2c = get_marker_pose(img, solver)
    transform = patch_target2base(c2b @ t2c)
    z_off = -0.045
    corn1 = np.array([0,0,z_off])
    corn2 = np.array([*TRAY_ID2_CENTER, z_off])
    capture_grid(
        corn1,
        corn2,
        robot,
        pose=np.eye(4),
        z_axis_rotations=z_angles,
        transform=transform,
        
    )
