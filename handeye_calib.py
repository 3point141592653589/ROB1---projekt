from pathlib import Path

import cv2 as cv
import numpy as np

from vision.marker_pose import CharucoPoseSolver, get_marker_pose


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
    K = np.load("./calibration_data/cam_params_final/K.npy")
    dist = np.load("./calibration_data/cam_params_final/dist.npy")
    calib_data = Path("./datasets/handeye_data/")
    out_dir = Path("./calibration_data/handeye_output_final/")
    method = CharucoPoseSolver(K, dist, (5, 5), 0.02, 0.015)
    n_data = len(list(calib_data.glob("*"))) // 3  # HACK: getting sick of this

    enhance = False
    vis = False

    R_gripper2base = []
    t_gripper2base = []
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
        target2cam = get_marker_pose(image, method)
        if target2cam is None:
            missed += 1
            undetected.append(i)
            continue

        R_t2c, t_t2c = homo_to_R_t(target2cam)
        R_target2cam.append(R_t2c)
        t_target2cam.append(t_t2c)

        gripper2base = np.load(calib_data / f"pos_{i}.npy")
        R_g2b, t_g2b = homo_to_R_t(gripper2base)
        R_gripper2base.append(R_g2b)
        t_gripper2base.append(t_g2b)

    Rt2g, tt2g, Rc2b, tc2b = cv.calibrateRobotWorldHandEye(
        R_gripper2base,
        t_gripper2base,
        R_target2cam,
        t_target2cam,
    )

    cam2base = np.eye(4)
    cam2base[:3, :3] = Rc2b
    cam2base[:3, 3] = tc2b.squeeze()
    target2gripper = np.eye(4)
    target2gripper[:3, :3] = Rt2g
    target2gripper[:3, 3] = tt2g.squeeze()
    out_dir.mkdir(exist_ok=True)
    np.save(out_dir / "cam2base.npy", cam2base)
    np.save(out_dir / "target2gripper.npy", target2gripper)

    print(f"missed {missed} photos")
    print(f"undetected: {undetected}")
    print(cam2base)
    print(target2gripper)


if __name__ == "__main__":
    main()
