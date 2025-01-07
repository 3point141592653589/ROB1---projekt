from .utils.tray import TRAY_CORNERS, get_tray_dict
from .vision import ArucoPoseSolver
import numpy as np
import cv2 as cv
from pathlib import Path

tray_imgs_dir = Path("./datasets/deska_5_6")
id0 = 5
id1 = 6

cam_param_dir = Path("./calibration_data/cam_params/")
K = np.load(cam_param_dir / "K.npy")
dist = np.load(cam_param_dir / "dist.npy")

for imgfile in tray_imgs_dir.glob("*"):
    img = cv.imread(str(imgfile))
    rvec, tvec = ArucoPoseSolver(get_tray_dict(5, 6), K, dist)(img)
    if rvec is None or tvec is None:
        continue
    imgp, _ = cv.projectPoints(np.array(TRAY_CORNERS), rvec, tvec, K, dist)
    cv.polylines(img, imgp.astype("uint32"), True, (255, 0, 0), 3)
