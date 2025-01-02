from __future__ import annotations

from typing import Protocol

import cv2 as cv
import numpy as np


class PoseMethod(Protocol):
    def __call__(
        self,
        img: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]: ...


class CharucoMethod(PoseMethod):
    def __init__(
        self,
        K,
        dist,
        board_size,
        square_size,
        aruco_size,
        aruco_dict_id=cv.aruco.DICT_4X4_100,
        legacypattern=True,
    ) -> None:
        ardict = cv.aruco.getPredefinedDictionary(aruco_dict_id)
        self.board = cv.aruco.CharucoBoard(
            board_size,
            square_size,
            aruco_size,
            ardict,
            None,
        )
        self.board.setLegacyPattern(legacypattern)
        self.detector = cv.aruco.CharucoDetector(self.board)
        self.K = K
        self.dist = dist

    def __call__(
        self,
        img: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        charucoCorners, charucoIds, markerCorners, markerIds = (
            self.detector.detectBoard(img)
        )
        if charucoCorners is None:
            return None, None
        objp, imgp = self.board.matchImagePoints(charucoCorners, charucoIds)
        cv.aruco.drawDetectedCornersCharuco(img, charucoCorners, charucoIds)
        if len(objp) < 4:
            return None, None
        method = cv.SOLVEPNP_IPPE if len(objp) > 4 else cv.SOLVEPNP_IPPE_SQUARE
        _, rvec, tvec = cv.solvePnP(
            np.array(objp),
            np.array(imgp),
            self.K,
            self.dist,
        )
        return rvec, tvec


class ArucoMethod(PoseMethod):
    def __init__(
        self,
        markers: dict[int, tuple[tuple[float, float], float]],
        K,
        dist,
        winsize=(7, 7),
        aruco_dictionary_id=cv.aruco.DICT_4X4_50,
    ) -> None:
        self.detector = cv.aruco.ArucoDetector(
            cv.aruco.getPredefinedDictionary(aruco_dictionary_id),
        )
        self.markers = markers
        self.K = K
        self.dist = dist
        self.winsize = winsize

    def __call__(
        self,
        img: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        corners, ids, rejected = self.detector.detectMarkers(img)
        if ids is None:
            return None, None
        objps = []
        pixelps = []
        for c, id in zip(corners, ids):
            if int(id) not in self.markers:
                continue
            rc = cv.cornerSubPix(
                img,
                c.astype("float32").reshape(4, 1, 2),
                winSize=self.winsize,  # Size of search window
                zeroZone=(-1, -1),  # Indicates no zero zone
                criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001),
            )
            center, size = self.markers[int(id)]
            for pixelp, signs in zip(
                rc.squeeze(),
                [(-1, 1), (1, 1), (1, -1), (-1, -1)],
            ):
                objp = np.zeros(3)
                objp[:2] = np.array(center) + size * np.array(signs) / 2
                objps.append(objp)
                pixelps.append(pixelp)
                cv.drawMarker(img, pixelp.astype("uint32"), (0, 0, 255))
        if not pixelps:
            return None, None
        method = cv.SOLVEPNP_IPPE_SQUARE if len(pixelps) == 1 else cv.SOLVEPNP_IPPE
        _, rvec, tvec = cv.solvePnP(
            np.array(objps),
            np.array(pixelps),
            self.K,
            self.dist,
            flags=method,
        )
        return rvec, tvec


def enhance_dark_image(img):
    # If color image, convert to gray
    if len(img.shape) == 3:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

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
    method: PoseMethod,
    vis=False,
) -> np.ndarray | None:
    rvec, tvec = method(img)
    if vis:
        cv.imshow("aruco corners", img)
        cv.waitKey(0)
    if rvec is None:
        return None
    R = cv.Rodrigues(rvec)[0]
    ret = np.eye(4)
    ret[:3, :3] = R
    ret[:3, 3] = tvec.squeeze()

    return ret


if __name__ == "__main__":
    img = cv.imread("./charuco_test.png")
    K = np.load("./cam_params/K.npy")
    dist = np.load("./cam_params/dist.npy")
    method = CharucoMethod(K, dist, (4, 4), 25, 18)
    x = get_camera_pose(img, method, vis=True)
    print(x)
