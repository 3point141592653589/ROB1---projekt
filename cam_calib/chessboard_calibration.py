from pathlib import Path

import cv2
import cv2 as cv
import numpy as np
from tqdm import tqdm


def get_image_errors(imgpoints, objpoints, rvecs, tvecs, K, dist):
    errors = []
    for i in range(len(imgpoints)):
        # Project points using current calibration
        projected_pts, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)

        # Calculate error for this image
        error = cv2.norm(imgpoints[i], projected_pts, cv2.NORM_L2) / len(projected_pts)
        errors.append(error)

    return np.array(errors)


def chessboard_calibration(
    chessboard_dims: tuple,
    imgs_path: str,
    no_dist=False,
    K_init=None,
    top_k=-1,
):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboard_dims[0] * chessboard_dims[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : chessboard_dims[0], 0 : chessboard_dims[1]].T.reshape(
        -1,
        2,
    )

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

            # termination criteria
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            cv.drawChessboardCorners(img, chessboard_dims, corners2, ret)
            cv.imshow("img", img)
            cv.waitKey(50)

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
    K_init = np.array([[4666, 0, 960], [0, 4666, 600], [0, 0, 1]], dtype="float64")
    err, K, dist, _, _ = chessboard_calibration(
        (9, 6),
        "./chessboard30",
        K_init=K_init,
        no_dist=False,
    )
    print(err)
    print(dist)
    print(K)
    dir = Path("./cam_params/")
    np.save(dir / "K.npy", K)
    np.save(dir / "dist.npy", dist)
