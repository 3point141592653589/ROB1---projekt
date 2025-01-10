from ctu_crs import CRS97


def init_robot(robot_class=CRS97, home=False):
    robot = robot_class()
    robot.reset_motors()
    robot.initialize(home=home)
    return robot
