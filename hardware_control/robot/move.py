import numpy as np
from ctu_crs.crs_robot import CRSRobot

from .rotations import euler2mat


def closest_q(qs, target_q):
    return min(qs, key=lambda x: float(np.linalg.norm(target_q - x)))


def closest_ik(robot: CRSRobot, pos_goal: np.ndarray):
    q_solutions = robot.ik(pos_goal)
    q_feasable = [q for q in q_solutions if robot.in_limits(q)]
    if not q_feasable:
        return None

    robot_q = robot.get_q()
    return closest_q(q_feasable, robot_q)


def move_closest_ik(robot: CRSRobot, pos_goal: np.ndarray):
    return move_rotated_ik(robot, pos_goal, [0])


def move_rotated_ik(robot: CRSRobot, pos_goal: np.ndarray, z_angles: list[float]):
    qs = []
    for th in z_angles:
        rotated_goal = pos_goal @ euler2mat([(th, "z")])
        q = closest_ik(robot, rotated_goal)
        if q is None:
            continue
        qs.append(q)
    if not qs:
        return False
    q = closest_q(qs, robot.get_q())
    robot.move_to_q(q)
    robot.wait_for_motion_stop()
    return True
