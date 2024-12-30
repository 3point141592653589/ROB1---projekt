from pathlib import Path

import cv2 as cv
import numpy as np
from tqdm import tqdm

from aruco_relative_pose import get_camera_pose

path = Path("./deska-1-2/")
K = np.load("./cam_params/K.npy")
dist = np.load("./cam_params/dist.npy")
# K = np.array([[4666, 0, 960], [0, 4666, 600], [0, 0, 1]], dtype="float64")
# dist = np.zeros(5)

vis = False
results = []
for f in tqdm(path.glob("*.bmp")):
    img = cv.imread(str(f))
    orig = get_camera_pose(img, {1: ((0, 0), 0.036)}, K, dist, vis=vis)[:3, 3]  # type: ignore
    second = get_camera_pose(img, {2: ((0.18, 0.14), 0.036)}, K, dist, vis=vis)[:3, 3]  # type: ignore
    res = np.linalg.norm(orig - second)
    results.append(res)
results = np.array(results)
print(f"mean: {np.mean(results)}\nmin: {np.min(results)}\nmax: {np.max(results)}")
