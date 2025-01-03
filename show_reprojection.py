from pathlib import Path

import cv2 as cv
import numpy as np
from ctu_crs import CRS97

cam2base = np.load("./handeye_output_patek/cam2base.npy")
target2gripper = np.load("./handeye_output_refined/target2gripper.npy")
K = np.load("./cam_calib/cam_params/K.npy")
dist = np.load("./cam_calib/cam_params/dist.npy")
data_dir = Path("./handeye_data/")
dh_offset = np.load("./handeye_output_patek/dh_offset.npy")
img_is = [50, 204, 208, 224]
# img_is = [40, 67, 200, 329, 350, 367]

board_size = (6, 6)
square_size = 0.02
objp = np.zeros((board_size[0] * board_size[1], 4), np.float32)
objp[:, :2] = (
    np.mgrid[0 : board_size[0], 0 : board_size[1]].T.reshape(
        -1,
        2,
    )
    * square_size
)
objp[:, 3] = np.ones(board_size[0] * board_size[1])

robot = CRS97(None)
robot.dh_offset = dh_offset

for i in img_is:
    image = cv.imread(str(data_dir / f"image_{i}.png"))
    angles = np.load(data_dir / f"joints_{i}.npy")
    g2b = robot.fk(angles)

    cam_points = (np.linalg.inv(cam2base) @ g2b @ target2gripper @ objp.T)[:3]
    projp, _ = cv.projectPoints(cam_points.T, np.zeros(3), np.zeros(3), K, dist)
    gripper_point = (np.linalg.inv(cam2base) @ g2b[:4, 3])[:3]
    gripper_proj, _ = cv.projectPoints(
        gripper_point.reshape((1, 3)),
        np.zeros(3),
        np.zeros(3),
        K,
        dist,
    )
    gx, gy = gripper_proj.ravel()
    cv.circle(image, (int(gx), int(gy)), 7, (255, 0, 0), -1)
    for point in projp:
        x, y = point.ravel()
        cv.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)  # Green circles

    cv.imshow("Projected Points", image)
    cv.waitKey(0)
    cv.destroyAllWindows()
