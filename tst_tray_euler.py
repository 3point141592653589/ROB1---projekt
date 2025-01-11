from pathlib import Path

import cv2 as cv
import numpy as np

from utils.tray import get_tray_dict
from vision.marker_pose import ArucoPoseSolver, get_marker_pose


def rotationMatrixToEulerAngles(R):
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = np.atan2(R[2, 1], R[2, 2])
        y = np.atan2(-R[2, 0], sy)
        z = np.atan2(R[1, 0], R[0, 0])
    else:
        x = np.atan2(-R[1, 2], R[1, 1])
        y = np.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


id0 = 7
id1 = id0 + 1

K = np.load("./cam_params2/K.npy")
dist = np.load("./calibration_data/cam_params/dist.npy")
img_dir = Path("./datasets/deska_7_8/")
for f in img_dir.glob("*"):
    print(str(f))
    img = cv.imread(str(f))

    t2c = get_marker_pose(
        img,
        ArucoPoseSolver(get_tray_dict(id0, id1), K, dist),
    )
    c2b = np.load("./handeye_output/cam2base.npy")
    print(np.rad2deg(rotationMatrixToEulerAngles((c2b @ t2c)[:3, :3])))
