from pathlib import Path

import cv2 as cv
import numpy as np

from utils.tray import TRAY_CORNERS, get_tray_dict
from vision import ArucoPoseSolver

id0 = 7
id1 = 8
tray_imgs_dir = Path(f"./datasets/deska_{id0}_{id1}")

cam_param_dir = Path("./calibration_data/cam_params/")
K = np.load(cam_param_dir / "K.npy")
dist = np.load(cam_param_dir / "dist.npy")

for imgfile in tray_imgs_dir.glob("*"):
    img = cv.imread(str(imgfile))
    rvec, tvec = ArucoPoseSolver(get_tray_dict(id0, id1), K, dist, (11, 11))(img)
    if rvec is None or tvec is None:
        continue
    imgp, _ = cv.projectPoints(np.array(TRAY_CORNERS), rvec, tvec, K, dist)
    cv.polylines(img, [imgp.astype("int32")], True, (255, 0, 0), 3)
    cv.namedWindow("deska", cv.WINDOW_NORMAL)
    cv.resizeWindow("deska", 960, 600)
    cv.imshow("deska", img)
    cv.waitKey(0)
