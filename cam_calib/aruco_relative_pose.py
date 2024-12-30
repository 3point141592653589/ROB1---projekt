from __future__ import annotations

import cv2 as cv
import numpy as np


def enhance_dark_image(img):
    # If color image, convert to gray
    if len(img.shape) == 3:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Method 1: CLAHE enhancement
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Method 2: Increase contrast/brightness
    alpha = 2.0  # Contrast control (1.0-3.0)
    beta = 30  # Brightness control (0-100)
    enhanced = cv.convertScaleAbs(enhanced, alpha=alpha, beta=beta)

    return enhanced


def detect_aruco_enhanced(img, detector):
    # Try detection on enhanced image
    enhanced = enhance_dark_image(img)
    corners, ids, rejected = detector.detectMarkers(enhanced)

    # If no detection, try different parameters
    if ids is None:
        # Try higher contrast
        enhanced = cv.convertScaleAbs(enhanced, alpha=3.0, beta=50)
        corners, ids, rejected = detector.detectMarkers(enhanced)

    return corners, ids, rejected, enhanced


def get_camera_pose(
    img: np.ndarray,
    markers: dict[int, tuple[tuple[float, float], float]],
    K: np.ndarray,
    dist: np.ndarray,
    aruco_dictionary_id=cv.aruco.DICT_4X4_50,
    method=cv.SOLVEPNP_IPPE_SQUARE,
    vis=False,
    enhance=False,
    winsize=(21, 21),
) -> np.ndarray | None:
    detector = cv.aruco.ArucoDetector(
        cv.aruco.getPredefinedDictionary(aruco_dictionary_id),
    )
    if enhance:
        corners, ids, rejected, enhanced = detect_aruco_enhanced(img, detector)
    else:
        corners, ids, rejected = detector.detectMarkers(img)
    if ids is None:
        return None
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    objps = []
    pixelps = []
    if vis:
        img_vis = enhanced if enhance else img.copy()
    for c, id in zip(corners, ids):
        if int(id) not in markers:
            continue
        rc = cv.cornerSubPix(
            gray,
            c.astype("float32").reshape(4, 1, 2),
            winSize=winsize,  # Size of search window
            zeroZone=(-1, -1),  # Indicates no zero zone
            criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )
        center, size = markers[int(id)]
        for pixelp, signs in zip(rc.squeeze(), [(-1, 1), (1, 1), (1, -1), (-1, -1)]):
            objp = np.zeros(3)
            objp[:2] = np.array(center) + size * np.array(signs) / 2
            objps.append(objp)
            pixelps.append(pixelp)
            if vis:
                cv.drawMarker(img_vis, pixelp.astype("uint32"), (0, 0, 255))
    if not pixelps:
        return None
    if vis:
        cv.imshow("aruco corners", img_vis)
        cv.waitKey(0)
    _, rvec, tvec = cv.solvePnP(
        np.array(objps),
        np.array(pixelps),
        K,
        dist,
        flags=method,
    )
    R = cv.Rodrigues(rvec)[0]
    ret = np.eye(4)
    ret[:3, :3] = R
    ret[:3, 3] = tvec.squeeze()

    return ret


if __name__ == "__main__":
    img = cv.imread("./im1.bmp")
    K = np.load("./cam_params/25mm2/K.npy")
    dist = np.load("./cam_params/25mm2/dist.npy")
    x = get_camera_pose(img, {1: ((0, 0), 0.0375)}, K, dist, vis=True)
    print(x)
