import numpy as np
from ctu_crs import CRS97
robot = CRS97()

robot.reset_motors()
#robot.initialize()
robot.initialize(home=False)

from save_conf import save_image_conf

save_image_conf(robot)