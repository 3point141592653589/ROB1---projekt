from pathlib import Path

import cv2 as cv
import numpy as np


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

    for fname in list(images):
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

            scale = 0.5
            resized_img = cv.resize(img, None, fx=scale, fy=scale, interpolation=cv.INTER_LINEAR)

            cv.imshow("img", resized_img)
            #cv.waitKey(250)
            cv.waitKey(0)

    cv.destroyAllWindows()
    return cv.calibrateCamera(
    objpoints, 
    imgpoints, 
    gray.shape[::-1], 
    None, 
    None, 
    flags=cv.CALIB_FIX_K3 | cv.CALIB_FIX_K4 | cv.CALIB_FIX_K5 | cv.CALIB_FIX_K6 | cv.CALIB_ZERO_TANGENT_DIST
)

if __name__ == "__main__":
    err, K, dist, _, _ = chessboard_calibration((9, 6), 1, "./chess30new")
    #err, K, dist, _, _ = chessboard_calibration((9, 6), 1, "./chessboard30")
    print(err)
    print(K)
    dir = Path("./cam_calib/cam_params")
    np.save(dir / "K.npy", K)
    np.save(dir / "dist.npy", dist)
