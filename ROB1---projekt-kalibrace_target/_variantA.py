#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2024-10-21
#     Author: Martin Cífka <martin.cifka@cvut.cz>
#


import numpy as np
import cv2 as cv
#from util import move_ik
from targrt_to_cam import target_to_cam

#from ctu_crs import CRS93
from ctu_crs import CRS97





def move_ik(robot, pos_goal):
    if robot is None:
        print(pos_goal)
        return None
    q_rad = robot.ik(pos_goal)
    q_radlim = [q for q in q_rad if robot.in_limits(q)]
    if not q_radlim:
        print("pose not reachable")
        return None

    robot_q = robot.get_q()
    q = min(q_radlim, key=lambda x: np.linalg.norm(robot_q - x))

    robot.move_to_q (q)
    robot.wait_for_motion_stop ( )
    return q




#robot = CRS93 (tty_dev="/dev/mars" )
#robot = CRS97 (tty_dev="/dev/mars" )
robot = None

#robot.initialize( home=False )
#robot.initialize( )

image = cv.imread("aruco.jpg")
#image = scan_picture()

cam_to_base = np.load("handeye_output/cam2base.npy")
if cam_to_base is None:
    print("No cam2base data")
    exit(1)


detector = cv.aruco.ArucoDetector(
        cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50),
    )

corners, ids, rejected = detector.detectMarkers(image)
if ids is None:
    print("No markers detected")
    exit(1)
min_id = min(ids)
targ_to_cam = target_to_cam(int(min_id[0]),image)

if targ_to_cam is None:
    print("No target detected")
    exit(1)

target_to_base =  targ_to_cam @ cam_to_base

i = 1
j = i+1
name = f"desky/positions_plate_0{i}-0{j}.csv"
data = np.loadtxt(name, delimiter=',', skiprows=1)

poz = np.eye(4)


for cube in data:
    poz[:2, 3] = cube/1000
    poz[2,3] = 0.01
    q = move_ik(robot, poz @ target_to_base)
    poz[2,3] = 0
    q = move_ik(robot, poz @ target_to_base)
    #robot.gripper.control_pozition(-800)
    poz[2,3] = 0.01
    q = move_ik(robot, poz @ target_to_base)
    q [0] = q[0]+np.pi/6
    robot.move_to_q (q)
    robot.wait_for_motion_stop ( )
    #robot.gripper.control_pozition(800)



# stop robot:
robot.soft_home ( )
robot.release ( )
robot.close ( )




