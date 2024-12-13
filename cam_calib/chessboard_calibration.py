from pathlib import Path

import cv2 as cv
import numpy as np
from tqdm import tqdm


def chessboard_calibration(chessboard_dims: tuple, square_size: float, imgs_path: str):
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboard_dims[0] * chessboard_dims[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : chessboard_dims[0], 0 : chessboard_dims[1]].T.reshape(
        -1,
        2,
    )
    objp *= square_size

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = Path(imgs_path).glob("*")

    for fname in tqdm(list(images)):
        img = cv.imread(str(fname))
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, chessboard_dims, None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)

            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            cv.drawChessboardCorners(img, chessboard_dims, corners2, ret)
            cv.imshow("img", img)
            cv.waitKey(500)

    cv.destroyAllWindows()
    return cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


if __name__ == "__main__":
    err, K, dist, _, _ = chessboard_calibration((8, 6), 0.025, "./chessboard 25x25mm")
    print(err)
    print(K)
    dir = Path("cam_params")
    np.save(dir / "K.npy", K)
    np.save(dir / "dist.npy", dist)
