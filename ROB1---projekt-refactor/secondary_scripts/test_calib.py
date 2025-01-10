from pathlib import Path

import cv2 as cv
import numpy as np

from aruco_relative_pose import get_camera_pose, ArucoMethod

path = Path("./cam_calib/deska_1_2/")
K = np.load("./cam_calib/cam_params/K.npy")
dist = np.load("./cam_calib/cam_params/dist.npy")
# K = np.array([[4666, 0, 960], [0, 4666, 600], [0, 0, 1]], dtype="float64")
# dist = np.zeros(5)

vis = False
results = []
for f in path.glob("*.bmp"):
    img = cv.imread(str(f))
    method = ArucoMethod({1: ((0, 0), 0.036)}, K, dist)
    orig = get_camera_pose(img.copy(), method, vis=vis)[:3, 3]  # type: ignore
    second_method = ArucoMethod({2: ((0.18, 0.14), 0.036)}, K, dist,)
    second = get_camera_pose(img.copy(), second_method, vis=vis)[:3, 3]  # type: ignore
    res = np.linalg.norm(orig - second)
    results.append(res)
results = np.array(results)
print(f"mean: {np.mean(results)}\nmin: {np.min(results)}\nmax: {np.max(results)}")
