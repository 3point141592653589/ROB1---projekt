from pathlib import Path

import cv2 as cv
import numpy as np

from cam_calib.aruco_relative_pose import get_camera_pose


def homo_to_R_t(C, inv=False):
    R = C[:3, :3]
    t = C[:3, 3]
    if inv:
        R = R.T
        t = -R @ t
    return R, t


def get_outlier_indices_iqr(arr, k=1.5):
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    low_bound = q1 - k * iqr
    high_bound = q3 + k * iqr
    return np.where((arr < low_bound) | (arr > high_bound))[0]  # outlier indices
    # Or for inliers: return np.where((arr >= low_bound) & (arr <= high_bound))[0]


def main():
    K = np.load("cam_calib/cam_params/K.npy")
    dist = np.load("cam_calib/cam_params/dist.npy")
    calib_data = Path("handeye_data3/")
    out_dir = Path("handeye_output/")
    n_data = len(list(calib_data.glob("*"))) // 4  # HACK: getting sick of this
    enhance = False
    vis = False

    R_base2gripper = []
    t_base2gripper = []
    R_target2cam = []
    t_target2cam = []

    missed = 0
    handpicked = []
    exclude_i = []
    undetected = []

    for i in range(n_data):
        if handpicked and i not in handpicked:
            continue
        if i in exclude_i:
            missed += 1
            continue
        image = cv.imread(str(calib_data / f"image_{i}.png"))
        if image is None:
            missed += 1
            continue
        if vis:
            print(i)
        target2cam = get_camera_pose(
            image,
            {2: ((0, 0), 0.036)},
            K,
            dist,
            method=cv.SOLVEPNP_IPPE_SQUARE,
            vis=vis,
            enhance=enhance,
            winsize=(11, 11),
        )
        if target2cam is None:
            missed += 1
            undetected.append(i)
            continue

        R_t2c, t_t2c = homo_to_R_t(target2cam)
        R_target2cam.append(R_t2c)
        t_target2cam.append(t_t2c)

        gripper2base = np.load(calib_data / f"pos_{i}.npy")
        R_b2g, t_b2g = homo_to_R_t(gripper2base, inv=True)
        R_base2gripper.append(R_b2g)
        t_base2gripper.append(t_b2g)

    R, t = cv.calibrateHandEye(
        R_base2gripper,
        t_base2gripper,
        R_target2cam,
        t_target2cam,
    )

    checksums = np.array(
        [
            t[2] + (-R.T @ g)[2]
            for g, t, R in zip(t_base2gripper, t_target2cam, R_base2gripper)
        ],
    )
    print(get_outlier_indices_iqr(checksums, 1.5))

    cam2base = np.eye(4)
    cam2base[:3, :3] = R
    cam2base[:3, 3] = t.squeeze()
    out_dir.mkdir(exist_ok=True)
    np.save(out_dir / "cam2base.npy", cam2base)

    print(f"missed {missed} photos")
    print(f"undetected: {undetected}")
    print(f"checksums: {checksums}")
    print(t)


if __name__ == "__main__":
    main()
