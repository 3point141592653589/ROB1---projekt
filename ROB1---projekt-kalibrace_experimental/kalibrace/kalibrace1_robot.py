#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2024-10-21
#     Author: Martin Cífka <martin.cifka@cvut.cz>
#
from pathlib import Path
import cv2
import numpy as np
from perception_utils import max_resize
from matplotlib import pyplot as plt


# OpenCV library for image processing
import cv2
# Our Basler camera interface
from basler_camera import BaslerCamera


#from ctu_crs import CRS93
from ctu_crs import CRS97





def to_homogeneous(x):
    return np.array([x[0], x[1], 1.0])


def from_homogeneous(x):
    return x[:2] / x[2]

def from_desk_to_image(v,H):
    return from_homogeneous(H @ to_homogeneous(v))


def scan_picture():
# take picture with camera

    camera: BaslerCamera = BaslerCamera()
 
    # Camera can be connected based on its' IP or name:
    # Camera for robot CRS 93
    #   camera.connect_by_ip("192.168.137.107")
    #   camera.connect_by_name("camera-crs93")
    # Camera for robot CRS 97
    #   camera.connect_by_ip("192.168.137.106")
    #   camera.connect_by_name("camera-crs97")
    camera.connect_by_name("camera-crs97")
 
    # Open the communication with the camera
    camera.open()
    # Set capturing parameters from the camera object.
    # The default parameters (set by constructor) are OK.
    # When starting the params should be send into the camera.
    camera.set_parameters()
    # Starts capturing images by camera.
    # If it is not done, the grab_image method starts it itself.
    camera.start()
 
    # Take one image from the camera
    img = camera.grab_image()
    # If the returned image has zero size,
    # the image was not captured in time.
    if (img is not None) and (img.size > 0):
        # Show the image in OpenCV
        cv2.namedWindow('Camera image', cv2.WINDOW_NORMAL)
        cv2.imshow('Camera image', img)
        #cv2.waitKey(0) ## no waiting
    else:
        print("The image was not captured.")
 
    # Close communication with the camera before finish.
    camera.close()
    return img

def main(robot):



    img = scan_picture() # take picture with camera


    #img = cv2.imread("aruco.jpg")  # load picture from file


    assert img is not None
    img = max_resize(img, max_width=1024)
    frame = img.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict)
    corners, ids, rejected = detector.detectMarkers(gray)

    n_markers = 2
    corners = np.array(corners).reshape(n_markers, 4, 2)

    # Sort the corners by the detected ids
    sort = np.argsort(ids.flatten())
    ids = ids[sort]
    corners = corners[sort]

    # Draw the 4 corners of the first detected marker
    plt.imshow(img[:, :, ::-1])
    #for i, c in zip(range(4), ["r", "g", "b", "y"]):
    #    plt.scatter(*corners[0, i], c=c)s

    # show some corners
    plt.scatter(*corners[0, 0], c="r")
    plt.scatter(*corners[0, 1], c="g")
    plt.scatter(*corners[1, 2], c="b")
    plt.scatter(*corners[1, 3], c="y")


    plt.show()

    # Estimate the homography
    # (using only the 4 corners of the first detected marker -> not precise)
    #src = np.array([[10, 10], [40, 10], [40, 40], [10, 40]], dtype=np.float32)
    #dst = corners[0]
    #H, _ = cv2.findHomography(src, dst)
    # TODO: 1. Get better estimate of H by using different points


    # used points 1 and 3 - diagonal
    #src = np.array([[10, 10], [40, 10], [287, 200], [257, 200]], dtype=np.float32)
    #dst = np.array([corners[0,0],corners[0,1],corners[3,2],corners[3,3]])


    # calibrate camera - a4
    #src = np.array([[10, 10], [40, 10], [40, 40], [10, 40], [257, 170], [287, 170], [287, 200], [257, 200]], dtype=np.float32)
    #dst = np.array([corners[0,0],corners[0,1],corners[0,2],corners[0,3],corners[3,0],corners[3,1],corners[3,2],corners[3,3]])
    
    # calibrate camera - a4 - origin in the middle of arc
    #src = np.array([[-15, -15], [15, -15], [15, 15], [-15, 15], [297-35-30, 210-35-30], [297-35, 210-35-30], [297-35, 210-35], [297-35-30, 210-35]], dtype=np.float32)
    #dst = np.array([corners[0,0],corners[0,1],corners[0,2],corners[0,3],corners[3,0],corners[3,1],corners[3,2],corners[3,3]])
    

   # calibrate camera - camera - origin in the middle of arc
    src = np.array([[-18, 18], [18, 18], [18, -18], [-18, -18], [180-18, 140+18], [180+18, 140+18], [180+18, 140-18], [180-18, 140-18]], dtype=np.float32)
    dst = np.array([corners[0,0],corners[0,1],corners[0,2],corners[0,3],corners[1,0],corners[1,1],corners[1,2],corners[1,3]])
    


    #src = np.array([[257, 170], [287, 170], [287, 200], [257, 200]], dtype=np.float32)
    #dst = corners[3]

    
    H, _ = cv2.findHomography(src, dst)



    # Project the corners of the A4 paper/centers of the markers to the image
    a_0 = [0, 0]
    a_paper = to_homogeneous(a_0)  # homogeneous coordinates on the A4 paper
    a_image = from_homogeneous(H @ a_paper)  # projective transformation
    #a_x = from_homogeneous(np.linalg.inv(H)@to_homogeneous(a_image))

    #b_0 = [297, 210]
    #b_0 = [257,160]
    b_0 = [180,140]
    b_paper = to_homogeneous(b_0)
    b_image = from_homogeneous(H @ b_paper)
    #b_x = from_homogeneous(np.linalg.inv(H)@to_homogeneous(b_image))


    
    b_sh = b_image-a_image
    th = np.arccos((b_0@b_sh)/(np.linalg.norm(b_0)*np.linalg.norm(b_sh)))

    


    plt.imshow(img[:, :, ::-1])
    plt.scatter(*a_image, c="r")
    plt.scatter(*b_image, c="g")
    plt.show()



   
    #q_rad = robot.get_q ( )
    #pose = robot.fk( q_rad )
    pose = np.array([[-1, 0, 0, 0.5],[0,1,0,0],[0,0,-1,0.1],[0,0,0,1]])

    d_base = [pose[0,3],pose[1,3]]
    thc = np.arccos(pose[0,0])

    d = [80,60]
    #d = [60,80]
    d_paper = to_homogeneous(d)
    d_cam = from_homogeneous(H @ d_paper)


    th_fin = thc + th
    rot = np.array([[np.cos(th_fin), -np.sin(th_fin)],[np.sin(th_fin), np.cos(th_fin)]])

    T = d_base - rot@ d_cam

    

    if __name__ == "__main__":
        #robot.release( )
        robot.soft_home ( )
        robot.close( )
    else:
        return T,rot,H




if __name__ == "__main__":
    #robot = CRS93 ( tty_dev=" /dev/mars " )
    robot = CRS97 ( tty_dev="/dev/mars" )
    robot.initialize (home=False)
    main(robot)

