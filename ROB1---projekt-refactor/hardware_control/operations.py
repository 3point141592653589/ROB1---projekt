from pathlib import Path

import numpy as np
from ctu_crs.crs_robot import CRSRobot

from hardware_control.robot.move import move_rotated_ik

from .io import save_image_conf
from .robot.rotations import euler2mat


def capture_grid(
    corn1: np.ndarray,
    corn2: np.ndarray,
    robot: CRSRobot,
    grid_shape=(5, 4),
    pose: np.ndarray | None = None,
    euler_rot: list[tuple[float, str]] | None = None,
    z_axis_rotations: list[float] | None = None,
    transform: np.ndarray | None = None,
    camera=None,
    output_dir="./datasets/handeye_data/",
):
    if z_axis_rotations is None:
        z_axis_rotations = [0]

    if pose is None:
        if euler_rot is None:
            e = "Arguments euler_rot and pose cannot both be None"
            raise RuntimeError(e)
        pose = euler2mat(euler_rot)

    if transform is None:
        transform = np.eye(4)

    pose[2, 3] = corn1[2]
    x1, y1 = corn1[:2]
    x2, y2 = corn2[:2]
    nx, ny = grid_shape
    Path(output_dir).mkdir(exist_ok=True)

    for x in np.linspace(x1, x2, nx):
        for y in np.linspace(y1, y2, ny):
            pose[:2, 3] = np.array([x, y])
            try:
                move_rotated_ik(robot, transform @ pose, z_axis_rotations)
                save_image_conf(robot, camera, dir=output_dir)
            except Exception as e:
                print(e)
                print("orientation IK failed:")
                print(pose)
