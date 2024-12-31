from pathlib import Path
from typing import Protocol

import cv2 as cv
import numpy as np


class CalibMethod(Protocol):
    def __call__(self, objpoints, imgpoints, img): ...


class CharucoMethod(CalibMethod):
    def __init__(
        self,
        board_size,
        square_size,
        aruco_size,
        aruco_dict_id=cv.aruco.DICT_4X4_100,
        legacyPattern=True,
    ) -> None:
        ardict = cv.aruco.getPredefinedDictionary(aruco_dict_id)
        self.board = cv.aruco.CharucoBoard(
            board_size,
            square_size,
            aruco_size,
            ardict,
            None,
        )
        self.board.setLegacyPattern(legacyPattern)
        self.detector = cv.aruco.CharucoDetector(self.board)

    def __call__(self, objpoints, imgpoints, img):
        charucoCorners, charucoIds, markerCorners, markerIds = (
            self.detector.detectBoard(img)
        )
        if charucoCorners is None:
            return
        objp, imgp = self.board.matchImagePoints(charucoCorners, charucoIds)
        objpoints.append(objp)
        imgpoints.append(imgp)
        cv.aruco.drawDetectedCornersCharuco(img, charucoCorners, charucoIds)


class ChessboardMethod(CalibMethod):
    def __init__(self, board_size) -> None:
        self.board_size = board_size
        self.objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0 : board_size[0], 0 : board_size[1]].T.reshape(
            -1,
            2,
        )

    def __call__(self, objpoints, imgpoints, img):
        ret, corners = cv.findChessboardCorners(img, self.board_size, None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(self.objp)

            # termination criteria
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            cv.drawChessboardCorners(img, self.board_size, corners2, ret)


def get_image_errors(imgpoints, objpoints, rvecs, tvecs, K, dist):
    errors = []
    for i in range(len(imgpoints)):
        # Project points using current calibration
        projected_pts, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)

        # Calculate error for this image
        error = cv.norm(imgpoints[i], projected_pts, cv.NORM_L2) / len(projected_pts)
        errors.append(error)

    return np.array(errors)


def chessboard_calibration(
    method: CalibMethod,
    imgs_path: str,
    no_dist=False,
    K_init=None,
    top_k=-1,
):
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = Path(imgs_path).glob("*")

    for fname in list(images):
        img = cv.imread(str(fname))
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        method(objpoints, imgpoints, gray)

        cv.imshow("img", gray)
        cv.waitKey(0)

    cv.destroyAllWindows()
    flags = cv.CALIB_FIX_K3
    if no_dist:
        flags += cv.CALIB_FIX_K1 + cv.CALIB_FIX_K2 + cv.CALIB_ZERO_TANGENT_DIST
    if K_init is not None:
        flags += cv.CALIB_USE_INTRINSIC_GUESS

    err, K, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        K_init,
        None,
        flags=flags,
    )

    errors = get_image_errors(imgpoints, objpoints, rvecs, tvecs, K, dist)

    top_k_indices = np.argsort(errors)[:top_k]

    filtered_imgpoints = [imgpoints[i] for i in top_k_indices]
    filtered_objpoints = [objpoints[i] for i in top_k_indices]

    return cv.calibrateCamera(
        filtered_objpoints,
        filtered_imgpoints,
        gray.shape[::-1],
        K,
        dist,
        flags=flags,
    )


if __name__ == "__main__":
    # K_init = np.array([[4666, 0, 960], [0, 4666, 600], [0, 0, 1]], dtype="float64")
    K_init = None
    method = CharucoMethod((10, 14), 20, 15)
    err, K, dist, _, _ = chessboard_calibration(
        method,
        "./phone_charuco/",
        K_init=K_init,
        no_dist=False,
    )
    print(err)
    print(dist)
    print(K)
    dir = Path("./cam_params/phone/")
    dir.mkdir(exist_ok=True)
    np.save(dir / "K.npy", K)
    np.save(dir / "dist.npy", dist)
