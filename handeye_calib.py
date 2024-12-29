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


def main():
    K = np.load("cam_calib/cam_params/25mm2/K.npy")
    dist = np.load("cam_calib/cam_params/25mm2/dist.npy")
    calib_data = Path("handeye_data2/")
    out_dir = Path("handeye_output/")
    n_data = len(list(calib_data.glob("*"))) // 4

    R_base2gripper = []
    t_base2gripper = []
    R_target2cam = []
    t_target2cam = []

    missed = 0
    exclude_i = [
        4,
        8,
        9,
        10,
        13,
        14,
        16,
        18,
        20,
        38,
        53,
        61,
        60,
        62,
        69,
        82,
        85,
        86,
        96,
        102,
        104,
        108,
        113,
        120,
        125,
        128,
        129,
        141,
        144,
        145,
        146,
        156,
        157,
    ]
    handpicked = [
        0,
        5,
        12,
        22,
        28,
        32,
        33,
        37,
        41,
        42,
        44,
        45,
        54,
        56,
        64,
        68,
        70,
        72,
        93,
        97,
        98,
        100,
        105,
        114,
        117,
        118,
        136,
        140,
        190,
        192,
        194,
        200,
        204,
    ]
    vis = False
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
            {1: ((0, 0), 0.0375)},
            K,
            dist,
            method=cv.SOLVEPNP_IPPE_SQUARE,
            vis=vis,
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

    cam2base = np.eye(4)
    cam2base[:3, :3] = R
    cam2base[:3, 3] = t.squeeze()
    out_dir.mkdir(exist_ok=True)
    np.save(out_dir / "cam2base.npy", cam2base)

    print(f"missed {missed} photos")
    print(f"undetected: {undetected}")
    print(t)


if __name__ == "__main__":
    main()
