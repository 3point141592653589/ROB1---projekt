from pathlib import Path

import cv2 as cv
import numpy as np
from ctu_crs import CRS97
from ctu_crs.crs_robot import CRSRobot
from scipy.optimize import least_squares

from vision.point_matching import CharucoPointMatcher


def optimize_reprojection(
    t2g: np.ndarray,  # WARNING: its actually gripper 2 target
    c2b: np.ndarray,
    angles_list: list[np.ndarray],
    objps: list[np.ndarray],
    imgps: list[np.ndarray],
    K: np.ndarray,
    dist: np.ndarray,
    robot: CRSRobot,
    min_points=8,
):
    # src_points and dst_points have shape (3, n)
    t2g_R = cv.Rodrigues(t2g[:3, :3])[0].squeeze()
    c2b_R = cv.Rodrigues(c2b[:3, :3])[0].squeeze()
    t2g_t = t2g[:3, 3]
    c2b_t = c2b[:3, 3]
    initial_guess = [*t2g_t, *t2g_R, *c2b_t, *c2b_R, *robot.dh_offset]

    def objective(params):
        (
            tx_t2g,
            ty_t2g,
            tz_t2g,
            r1_t2g,
            r2_t2g,
            r3_t2g,
            tx_c2b,
            ty_c2b,
            tz_c2b,
            r1_c2b,
            r2_c2b,
            r3_c2b,
            q0,
            q1,
            q2,
            q3,
            q4,
            q5,
        ) = params

        _t2g = np.eye(4)
        _c2b = np.eye(4)
        _t2g[:3, :3] = cv.Rodrigues(np.array([r1_t2g, r2_t2g, r3_t2g]))[0].squeeze()
        _t2g[:3, 3] = np.array([tx_t2g, ty_t2g, tz_t2g])
        _c2b[:3, :3] = cv.Rodrigues(np.array([r1_c2b, r2_c2b, r3_c2b]))[0].squeeze()
        _c2b[:3, 3] = np.array([tx_c2b, ty_c2b, tz_c2b])

        robot.dh_offset = np.array([q0, q1, q2, q3, q4, q5])
        errs = []

        for angles, objp, imgp in zip(angles_list, objps, imgps):
            if objp.shape[0] < min_points:
                continue
            objp_homo = np.vstack((objp.squeeze().T, np.ones(len(objp))))
            g2b = robot.fk(angles)
            cam_points = (np.linalg.inv(_c2b) @ g2b @ _t2g @ objp_homo)[:3]
            projp, _ = cv.projectPoints(cam_points.T, np.zeros(3), np.zeros(3), K, dist)
            err = np.linalg.norm(projp.squeeze() - imgp.squeeze(), axis=1)
            errs.extend(err)

        return np.array(errs)

    result = least_squares(
        objective,
        initial_guess,
    )
    (
        tx_t2g,
        ty_t2g,
        tz_t2g,
        r1_t2g,
        r2_t2g,
        r3_t2g,
        tx_c2b,
        ty_c2b,
        tz_c2b,
        r1_c2b,
        r2_c2b,
        r3_c2b,
        q0,
        q1,
        q2,
        q3,
        q4,
        q5,
    ) = result.x

    print(result.cost)

    cam2base = np.eye(4)
    cam2base[:3, :3] = cv.Rodrigues(np.array([r1_c2b, r2_c2b, r3_c2b]))[0].squeeze()
    cam2base[:3, 3] = np.array([tx_c2b, ty_c2b, tz_c2b])

    target2gripper = np.eye(4)
    target2gripper[:3, :3] = cv.Rodrigues(np.array([r1_t2g, r2_t2g, r3_t2g]))[
        0
    ].squeeze()
    target2gripper[:3, 3] = np.array([tx_t2g, ty_t2g, tz_t2g])

    return cam2base, target2gripper, np.array([q0, q1, q2, q3, q4, q5])


def constrained_orthogonal_procrustes(
    coord_from: list[np.ndarray],
    coord_to: list[np.ndarray],
):
    x = np.hstack(coord_to)
    y = np.hstack(coord_from)
    x_c = x.mean(1, keepdims=True)
    y_c = y.mean(1, keepdims=True)
    x_hat = x - x_c
    y_hat = y - y_c

    M = y_hat @ x_hat.T
    U, S, Vh = np.linalg.svd(M, compute_uv=True)

    if np.linalg.det(U @ Vh) < 0:
        S_ = np.eye(S.shape[0])
        S_[-1, -1] = -1
        R = U @ S_ @ Vh
    else:
        R = U @ Vh
    t = y_c - R @ x_c
    ret = np.eye(4)
    ret[:3, :3] = R
    ret[:3, 3] = t.squeeze()
    return ret


if __name__ == "__main__":
    cam2base = np.load("./calibration_data/handeye_output/cam2base.npy")
    target2gripper = np.load("./calibration_data/handeye_output/target2gripper.npy")
    K = np.load("./calibration_data/cam_params3/K.npy")
    dist = np.load("./calibration_data/cam_params3/dist.npy")
    target_method = CharucoPointMatcher((5, 5), 0.020, 0.015)
    data_dir = Path("./datasets/handeye_data/")
    out_dir = Path("./calibration_data/handeye_output_refined")
    out_dir.mkdir(exist_ok=True)
    n_data = len(list(data_dir.glob("*"))) // 3  # HACK: getting sick of this
    objps, imgps, angles_list = [], [], []
    for i in range(n_data):
        image = cv.imread(str(data_dir / f"image_{i}.png"))
        if image is None or not target_method(objps, imgps, image):
            continue
        angles_list.append(np.load(data_dir / f"joints_{i}.npy"))
    cam2base_refined, target2gripper_refined, dh_offset = optimize_reprojection(
        target2gripper,
        cam2base,
        angles_list,
        objps,
        imgps,
        K,
        dist,
        CRS97(None),
    )
    print(
        "cam2base",
        "refined",
        cam2base_refined,
        "original",
        cam2base,
        "target2gripper",
        target2gripper_refined,
        "original",
        target2gripper,
        "offset",
        np.rad2deg(dh_offset),
        sep="\n",
    )
    np.save(out_dir / "cam2base.npy", cam2base_refined)
    np.save(out_dir / "target2gripper.npy", target2gripper_refined)
    np.save(out_dir / "dh_offset.npy", dh_offset)
