import numpy as np
from ctu_crs import CRS97
robot = CRS97()

robot.reset_motors()
robot.initialize(home=False)
#robot.soft_home ( )

robot.gripper.control_position(-100)
#poz = np.array([[0, 0, 1, 0.7],[0,1,0,0],[-1,0,0,0.5],[0,0,0,1]])
#poz = np.array([[0, 0, 1, 0.7],[1,0,0,0],[0,1,0,0.5],[0,0,0,1]]) #kolem x

#deska grip
#poz = np.array([[-1, 0, 0, 0.5],[0,1,0,0],[0,0,-1,0.235],[0,0,0,1]])

#deska lift
#poz = np.array([[-1, 0, 0, 0.5],[0,1,0,0],[0,0,-1,0.35],[0,0,0,1]])

#deska flip
poz = np.array([[0,1,0,0.5], [-1,0,0,0.3], [0,0,1,0.35]])

#poz = robot.fk(np.radians([0,0,0,0,0,0]))

#robot.move_to_q (np.radians([0,-45,-45,0,0,0]))
q_rad = robot.ik(poz)
q_radlim = [q for q in q_rad if robot.in_limits(q)]

#q_deg = np.array( [ 0 , -45, -45, 0 , 0 , 0 ] )
#q_rad = np.deg2rad (q_deg)
robot.move_to_q (q_radlim[0])
robot.wait_for_motion_stop ( )
print(robot.in_motion( ))
q_rad = robot.get_q( )
q_deg = np.rad2deg(q_rad)
print("Position␣[deg]:" , q_deg )

robot.release()