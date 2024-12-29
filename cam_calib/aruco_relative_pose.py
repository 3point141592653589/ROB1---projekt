from __future__ import annotations

import cv2 as cv
import numpy as np


def get_camera_pose(
    img: np.ndarray,
    markers: dict[int, tuple[tuple[float, float], float]],
    K: np.ndarray,
    dist: np.ndarray,
    aruco_dictionary_id=cv.aruco.DICT_4X4_50,
    method=cv.SOLVEPNP_IPPE_SQUARE,
    vis=False,
) -> np.ndarray | None:
    detector = cv.aruco.ArucoDetector(
        cv.aruco.getPredefinedDictionary(aruco_dictionary_id),
    )
    corners, ids, rejected = detector.detectMarkers(img)
    if ids is None:
        return None
    objps = []
    pixelps = []
    if vis:
        img_vis = img.copy()
    for c, id in zip(corners, ids):
        if int(id) not in markers:
            continue
        center, size = markers[int(id)]
        for pixelp, signs in zip(c.squeeze(), [(-1, 1), (1, 1), (1, -1), (-1, -1)]):
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
