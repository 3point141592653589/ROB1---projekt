#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_camera.py: Experimental scripts for testing
image acquisition.
"""

__author__ = "Pavel Krsek"
__maintainer__ = "Pavel Krsek"
__email__ = "pavel.krsek@cvut.cz"
__copyright__ = "Copyright \xa9 2023 RMP, CIIRC CVUT in Prague\nAll Rights Reserved."
__license__ = "Use for lectures B3B33ROB1"
__version__ = "1.0"
__date__ = "2024/10/30"
__status__ = "Development"
__credits__ = []
__all__ = []

#: OpenCV library for image processing
import cv2
#: Our Basler camera interface
from basler_camera import BaslerCamera


def main():
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
        cv2.waitKey(0)
    else:
        print("The image was not captured.")

    # Close communication with the camera before finish.
    camera.close()


if __name__ == '__main__':
    main()