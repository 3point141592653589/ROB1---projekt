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





def to_homogeneous(x):
    return np.array([x[0], x[1], 1.0])


def from_homogeneous(x):
    return x[:2] / x[2]


def main():
    img = cv2.imread("aruco.jpg")
    assert img is not None
    img = max_resize(img, max_width=1024)
    frame = img.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict)
    corners, ids, rejected = detector.detectMarkers(gray)

    n_markers = 4
    corners = np.array(corners).reshape(n_markers, 4, 2)

    # Sort the corners by the detected ids
    sort = np.argsort(ids.flatten())
    ids = ids[sort]
    corners = corners[sort]

    # Draw the 4 corners of the first detected marker
    plt.imshow(img[:, :, ::-1])
    #for i, c in zip(range(4), ["r", "g", "b", "y"]):
    #    plt.scatter(*corners[0, i], c=c)s

    plt.scatter(*corners[0, 0], c="r")
    plt.scatter(*corners[0, 1], c="g")
    plt.scatter(*corners[3, 2], c="b")
    plt.scatter(*corners[3, 3], c="y")


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

    src = np.array([[10, 10], [40, 10], [40, 40], [10, 40], [257, 170], [287, 170], [287, 200], [257, 200]], dtype=np.float32)
    dst = np.array([corners[0,0],corners[0,1],corners[0,2],corners[0,3],corners[3,0],corners[3,1],corners[3,2],corners[3,3]])
    
    #src = np.array([[257, 170], [287, 170], [287, 200], [257, 200]], dtype=np.float32)
    #dst = corners[3]

    
    H, _ = cv2.findHomography(src, dst)



    # Project the corners of the A4 paper to the image
    a_0 = [0, 0]
    a_paper = to_homogeneous(a_0)  # homogeneous coordinates on the A4 paper
    a_image = from_homogeneous(H @ a_paper)  # projective transformation
    #a_x = from_homogeneous(np.linalg.inv(H)@to_homogeneous(a_image))

    b_0 = [297, 210]
    b_paper = to_homogeneous(b_0)
    b_image = from_homogeneous(H @ b_paper)
    #b_x = from_homogeneous(np.linalg.inv(H)@to_homogeneous(b_image))



    b_sh = b_image-a_image
    th = np.arccos((b_0@b_sh)/(np.linalg.norm(b_0)*np.linalg.norm(b_sh)))

    rot = np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]])
    btest2 = rot@np.array(b_0)
    btest3 = btest2/np.linalg.norm(btest2)
    bt = b_sh[0]/btest3[0]*btest3[1]+a_image[1]
    bt = b_sh[0]/btest2[0]*btest2[1]+a_image[1]

    th = -th
    rot = np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]])
    #rot = np.array([[np.cos(th),np.sin(th)],[np.sin(th),-np.cos(th)]])#preklopeni y

    btest2 = rot@np.array(b_sh)
    btest3 = btest2/np.linalg.norm(btest2)
    bt = 297/btest3[0]*btest3[1]

    plt.imshow(img[:, :, ::-1])
    plt.scatter(*a_image, c="r")
    plt.scatter(*b_image, c="g")
    plt.show()

    # TODO: 2. Use the homography H to find the coordinates of the point
    # on the paper [mm], in the coordinate system printed on the paper
    #
    # A4 paper - 297x210 mm
    # Margin - 10 mm
    # Marker size - 30 mm


if __name__ == "__main__":
    main()
