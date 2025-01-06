from ctu_crs import CRS97


def init_robot():
    robot = CRS97()
    robot.reset_motors()
    robot.initialize(home=False)
    return robot
