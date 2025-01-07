from pathlib import Path

import cv2 as cv
import numpy as np
from ctu_crs import CRS97

cam2base = np.load("./handeye_output_refined5/cam2base.npy")
dh_offset = np.load("./handeye_output_refined5/dh_offset.npy")
cam2base2 = np.load("./handeye_output_patek/cam2base.npy")
dh_offset2 = np.load("./handeye_output_patek/dh_offset.npy")
robot = CRS97(None)
robot2 = CRS97(None)
robot.dh_offset = dh_offset
robot2.dh_offset = dh_offset2
K = np.load("./cam_calib/cam_params/K.npy")
dist = np.load("./cam_calib/cam_params/dist.npy")
data_dir = Path("./handeye_data5/")
data_dir2 = Path("./handeye_data_patek/")

for data in (data_dir, data_dir2):
    n_data = len(list(data.glob("*"))) // 4  # HACK: getting sick of this
    dists = []
    for i in range(n_data):
        image = cv.imread(str(data_dir / f"image_{i}.png"))
        if image is None:
            continue
        angles = np.load(data_dir / f"joints_{i}.npy")
        g2b = robot.fk(angles)
        g2b2 = robot2.fk(angles)
        dists.append(
            np.linalg.norm(
                np.linalg.inv(cam2base) @ g2b[:4, 3]
                - np.linalg.inv(cam2base2) @ g2b2[:4, 3],
            ),
        )
    dists = np.array(dists)
    print(np.mean(dists))
