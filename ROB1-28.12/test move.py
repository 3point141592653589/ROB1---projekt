import numpy as np
from ctu_crs import CRS97
robot = CRS97()
from move_utils import rot_matrix
from util import move_ik

robot.reset_motors()
robot.initialize(home=False)
#robot.initialize()
#robot.soft_home ( )


#robot.gripper.control_position(1000)
#q_rad = robot.get_q( )
#q_rad[0] = q_rad[0]+np.pi/3
#robot.move_to_q (q_rad)
#robot.release()




#robot.gripper.control_position(500)
# poz = np.array([[0, 0, 1, 0.7],[0,1,0,0],[-1,0,0,0.5],[0,0,0,1]])
#poz = np.array([[0, 0, 1, 0.7],[1,0,0,0],[0,1,0,0.5],[0,0,0,1]]) #kolem x

#deska grip
#poz = np.array([[-1, 0, 0, 0.5],[0,1,0,0],[0,0,-1,0.235],[0,0,0,1]])

# poz = np.array([[-1, 0, 0, 0.34],[0,1,0,0],[0,0,-1,0.235],[0,0,0,1]])

#deska lift
#poz = np.array([[-1, 0, 0, 0.5],[0,1,0,0],[0,0,-1,0.35],[0,0,0,1]])

#deska flip
#poz = np.array([[0,1,0,0.5], [-1,0,0,0.3], [0,0,1,0.35]])

#poz = robot.fk(np.radians([0,0,0,0,0,0]))

#poz = robot.fk(robot.get_q())
#poz[3, 0] += 0.01

poz = np.eye(4)
poz[:3, :3] = rot_matrix(np.pi, "y") @ rot_matrix(-np.pi/4, "z")
poz[:3, 3] = np.array([0.46-0.155, -(0.22-0.28), 0.235])

#robot.move_to_q (np.radians([0,-45,-45,0,0,0]))
#q_rad = robot.ik(poz)
#q_radlim = [q for q in q_rad if robot.in_limits(q)]

#q_deg = np.array( [ 0 , -45, -45, 0 , 0 , 0 ] )
#q_rad = np.deg2rad (q_deg)

move_ik(robot, poz)

q_rad = robot.get_q( )
q_deg = np.rad2deg(q_rad)
print("Position␣[deg]:" , q_deg )

#robot.release()