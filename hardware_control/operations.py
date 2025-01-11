import time
from pathlib import Path
from typing import Literal

import numpy as np
from ctu_crs.crs_robot import CRSRobot

from .io import save_image_conf
from .robot.move import move_closest_ik as move
from .robot.move import move_rotated_ik
from .robot.rotations import euler2mat

GRIPPER_VALS = {"grab": -800, "place": 600}
Z_NEAR = 0.08
Z_PLACE = 0.04
MIDPOINT = euler2mat([(np.pi, "y")])
MIDPOINT[:3, 3] = [0.5, 0, 0.3]
TRAY_OFFSET_X = 0.0005
TRAY_OFFSET_Y = 0.006


def move_cubes(
    robot: CRSRobot,
    camera,
    targ1_to_base_function,
    targ2_to_base_function,
    cube_data1: np.ndarray,
    cube_data2: np.ndarray,
):
    for i in range(len(cube_data1)):
        if i:
            q = robot.get_q()
            q[0] += np.pi / 6
            robot.move_to_q(q)
            robot.wait_for_motion_stop()
        image = camera.grab_image()
        targ1_to_base = targ1_to_base_function(image)
        targ2_to_base = targ2_to_base_function(image)
        grab_cube(robot, targ1_to_base, cube_data1[i], "grab")
        move(robot, MIDPOINT)
        grab_cube(robot, targ2_to_base, cube_data2[i], "place")
        move(robot, MIDPOINT)


def grab_cube(
    robot: CRSRobot,
    targ_to_base: np.ndarray,
    cube: list,
    mode: Literal["grab", "place"],
):
    poz = euler2mat([(np.pi, "y")])

    poz[0, 3] = cube[0] / 1000 - TRAY_OFFSET_X
    poz[1, 3] = cube[1] / 1000 - TRAY_OFFSET_Y
    # poz[1, 3] *= 1.09
    poz[2, 3] = Z_NEAR
    move(robot, targ_to_base @ poz)

    poz[2, 3] = Z_PLACE
    move(robot, targ_to_base @ poz)
    if robot is not None:
        robot.gripper.control_position(GRIPPER_VALS[mode])
        time.sleep(1.5)
    poz[2, 3] = Z_NEAR
    move(robot, targ_to_base @ poz)


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
        pose = np.eye(4)
        pose[:3, :3] = euler2mat(euler_rot)

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
