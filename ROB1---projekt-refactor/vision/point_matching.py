from typing import Protocol

import cv2 as cv
import numpy as np


class PointMatcher(Protocol):
    def __call__(self, objpoints, imgpoints, img) -> bool: ...


class CharucoPointMatcher(PointMatcher):
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

    def __call__(self, objpoints, imgpoints, img) -> bool:
        charucoCorners, charucoIds, markerCorners, markerIds = (
            self.detector.detectBoard(img)
        )
        if charucoCorners is None:
            return False
        objp, imgp = self.board.matchImagePoints(charucoCorners, charucoIds)
        objpoints.append(objp)
        imgpoints.append(imgp)
        cv.aruco.drawDetectedCornersCharuco(img, charucoCorners, charucoIds)
        return True


class ChessboardPointMatcher(PointMatcher):
    def __init__(self, board_size) -> None:
        self.board_size = board_size
        self.objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0 : board_size[0], 0 : board_size[1]].T.reshape(
            -1,
            2,
        )

    def __call__(self, objpoints, imgpoints, img) -> bool:
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
        return ret
