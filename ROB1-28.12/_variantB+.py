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
ids = np.array(ids)
ids = ids[:,0]

ids = [ids[0],ids[1],5,6]


sorted_ids = np.sort(ids)
if not (len(sorted_ids) == 4):
    print("wrong number of markers detected")
    exit(1)


while True:
    print("Choose bloks source desk: 0:(IDs:",sorted_ids[0],",",sorted_ids[1],"),1:(IDs:",sorted_ids[2],",",sorted_ids[3],")")
    num_input = input()
    if num_input.isdigit():
        direction = int(num_input)
        break
    else:
        print("It's not a number")
if not direction == 0:
    direction = 1
targ1_to_cam = target_to_cam(sorted_ids[direction],image)
targ2_to_cam = target_to_cam(sorted_ids[2*(1-direction)],image)


if targ1_to_cam is None or targ2_to_cam is None:
    print("No target detected")
    exit(1)



targ1_to_base = cam_to_base @ targ1_to_cam
targ2_to_base = cam_to_base @ targ2_to_cam
#print(targ_to_base-np.dot(cam_to_base, targ_to_cam))


if robot is None:
    move(robot, targ1_to_base,color='r')

data = np.array([0,0])
h = 0
for file in [direction,1-direction]:
    i = sorted_ids[2*file]
    j = i+1
    name = f"desky/positions_plate_0{i}-0{j}.csv"
    data[h] = np.loadtxt(name, delimiter=',', skiprows=1)
    h = 1

#poz = np.eye(4)


for i in range(len(data[0])):
    grab_cube(robot, targ1_to_base, data[0,i], -800)
    grab_cube(robot, targ2_to_base, data[1,i], 800)


if robot is not None:
    robot.soft_home ( )
    robot.release ( )
    robot.close ( )

else:
    plt.show()






