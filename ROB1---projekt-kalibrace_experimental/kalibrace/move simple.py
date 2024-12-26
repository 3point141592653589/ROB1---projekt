#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2024-10-21
#     Author: Martin Cífka <martin.cifka@cvut.cz>
#


import numpy as np

from kalibrace1_robot import main as calibration
from kalibrace1_robot import from_desk_to_image

#from ctu_crs import CRS93
from ctu_crs import CRS97


def move_desk(v_desk,T,rot,H):
    v_camera = from_desk_to_image([v_desk[0],v_desk[1]],H)
    v_base_2D = rot @ v_camera + T
    T_base_3d = np.eye(4)
    T_base_3d [:2,:2] = rot
    T_base_3d [3,:2] = v_base_2D
    T_base_3d [3,2] = v_desk[2]
    return T_base_3d


#robot = CRS93 (tty_dev="/dev/mars" )
robot = CRS97 (tty_dev="/dev/mars" )



#robot.initialize( home=False )
robot.initialize( )



# stop robot:
#robot.release ( )


# move

q_deg = np.array( [ 0 , -45, -45, 0 , 0 , 0 ] )
q_rad = np.deg2rad (q_deg)
robot.move_to_q (q_rad)
robot.wait_for_motion_stop ( )
print(robot.in_motion( ))
q_rad = robot.get_q( )
q_deg = np.rad2deg(q_rad)
print("Position␣[deg]:" , q_deg )

# end of code
robot.soft_home( )
robot.close( )


#kynematic

#pose = robot.fk( [ 0 , 0 , 0 , 0 , 0 , 0 ] )
T,rot,H = calibration(robot)


qs = robot.ik(move_desk([0,0,30],T,rot,H))
robot.move_to_q (qs)
robot.wait_for_motion_stop ( )
print(robot.in_motion( ))
q_rad = robot.get_q( )
q_deg = np.rad2deg(q_rad)
print("Position␣[deg]:" , q_deg )