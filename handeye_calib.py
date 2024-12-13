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
    K = np.load("./cam_calib/cam_params/K.npy")
    dist = np.load("./cam_calib/cam_params/dist.npy")
    calib_data = Path("./handeye_data/")
    out_dir = Path("./handeye_output/")
    n_data = len(list(calib_data.glob("*"))) // 3

    R_base2gripper = []
    t_base2gripper = []
    R_target2cam = []
    t_target2cam = []

    missed = 0

    for i in range(n_data):
        if not i:
            continue
        image = cv.imread(str(calib_data / f"image_{i}.png"))
        target2cam = get_camera_pose(
            image,
            {2: ((0, 0), 0.036)},
            K,
            dist,
            method=cv.SOLVEPNP_IPPE_SQUARE,
        )
        if target2cam is None:
            missed += 1
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
    print(t)


if __name__ == "__main__":
    main()
