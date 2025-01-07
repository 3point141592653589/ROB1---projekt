import numpy as np

from ctu_crs.crs_robot import CRSRobot


def move_ik(robot: CRSRobot, pos_goal: np.ndarray):
    q_rad = robot.ik(pos_goal)
    q_radlim = [q for q in q_rad if robot.in_limits(q)]
    if not q_radlim:
        print("pose not reachable")
        return

    robot_q = robot.get_q()
    q = min(q_radlim, key=lambda x: float(np.linalg.norm(robot_q - x)))

    robot.move_to_q(q)
    robot.wait_for_motion_stop()
