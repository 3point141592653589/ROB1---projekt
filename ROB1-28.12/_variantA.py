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
#from move_rob import move_ik as move
from move_viz import plot_transform as move

#from ctu_crs import CRS93
from ctu_crs import CRS97


import matplotlib.pyplot as plt


def grab_cube(robot, targ_to_base, cube, gripper):
    poz[:2, 3] = cube/1000
    poz[2,3] = 0.03
    q = move(robot,targ_to_base @ poz)
    '''
    poz[2,3] = 0.01
    q = move(robot, targ_to_base @ poz)
    if robot is not None:
        robot.gripper.control_pozition(gripper)
    poz[2,3] = 0.03
    q = move(robot, targ_to_base @ poz)
    #'''
    return q





#robot = CRS93 (tty_dev="/dev/mars" )
#robot = CRS97 (tty_dev="/dev/mars" )
robot = None

#robot.initialize( home=False )
#robot.initialize( )

image = cv.imread("aruco\Image__2024-12-26__13-25-35.bmp")
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

#cam_to_base[:3,:3] = np.eye(3)


targ_to_base = cam_to_base @ targ_to_cam
#print(targ_to_base-np.dot(cam_to_base, targ_to_cam))

'''
cam_to_base = np.linalg.inv(cam_to_base)
targ_to_base = np.dot(cam_to_base, targ_to_cam)
#targ_to_base = np.linalg.inv(targ_to_base)
'''

if robot is None:
    move(robot, targ_to_base,color='r')

i = int(min_id[0])
j = i+1
name = f"desky/positions_plate_0{i}-0{j}.csv"
data = np.loadtxt(name, delimiter=',', skiprows=1)

poz = np.eye(4)


for cube in data:
    grab_cube(robot, targ_to_base, cube, -800)

    '''
    if robot is not None:
        q [0] = q[0]+np.pi/6
        robot.move_to_q (q)
        robot.wait_for_motion_stop ( )
        robot.gripper.control_pozition(800)
    #'''

if robot is not None:
    robot.soft_home ( )
    robot.release ( )
    robot.close ( )

else:
    plt.show()






