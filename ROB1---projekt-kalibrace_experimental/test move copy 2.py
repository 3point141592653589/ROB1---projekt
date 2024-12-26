import numpy as np
from ctu_crs import CRS97
robot = CRS97()

robot.reset_motors()
robot.initialize(home=False)
#robot.initialize()
#robot.soft_home ( )


#robot.gripper.control_position(1000)
#q_rad = robot.get_q( )
#q_rad[0] = q_rad[0]+np.pi/3
#robot.move_to_q (q_rad)
#robot.release()



q = robot.get_q()
q[-1] = 0.5*np.pi/2

robot.move_to_q (q)
robot.wait_for_motion_stop ( )
print(robot.in_motion( ))
q_rad = robot.get_q( )
q_deg = np.rad2deg(q_rad)
print("Position␣[deg]:" , q_deg )

robot.release()