from pathlib import Path

import cv2 as cv
import numpy as np
from .vision import PointMatcher, CharucoPointMatcher


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
    method: PointMatcher,
    imgs_path: str,
    no_dist=False,
    K_init=None,
    top_k=-1,
):
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = Path(imgs_path).glob("*")
    if not images:
        e = "No images in path"
        raise RuntimeWarning(e)

    for fname in list(images):
        img = cv.imread(str(fname))
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        method(objpoints, imgpoints, gray)

        cv.imshow("img", gray)
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
    # K_init = np.array([[4666, 0, 960], [0, 4666, 600], [0, 0, 1]], dtype="float64")
    K_init = None
    method = CharucoPointMatcher((10, 14), 20, 15)
    err, K, dist, _, _ = chessboard_calibration(
        method,
        "./charuco20_15/",
        K_init=K_init,
    )
    print(err)
    print(dist)
    print(K)
    dir = Path("./cam_params/")
    dir.mkdir(exist_ok=True)
    np.save(dir / "K.npy", K)
    np.save(dir / "dist.npy", dist)
