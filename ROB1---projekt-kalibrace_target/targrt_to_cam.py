from pathlib import Path

import cv2 as cv
import numpy as np

from cam_calib.aruco_relative_pose import get_camera_pose


# Our Basler camera interface
from basler_camera import BaslerCamera


def homo_to_R_t(C, inv=False):
    R = C[:3, :3]
    t = C[:3, 3]
    if inv:
        R = R.T
        t = -R @ t
    return R, t


def target_to_cam(aru_id,image):
    K = np.load("cam_calib/cam_params/K.npy")
    dist = np.load("cam_calib/cam_params/dist.npy")
    #cam2base = np.load("handeye_output/cam2base.npy")





    if image is None:
        return None
    target2cam = get_camera_pose(
        image,
        {aru_id: ((0, 0), 0.03)},
        K,
        dist,
        method=cv.SOLVEPNP_IPPE_SQUARE,
    )
    if target2cam is None:
        return None

    #R_t2c, t_t2c = homo_to_R_t(target2cam)
    #'''
    point = np.eye(4)
    point[:3, 3] = [0.18,0.14,0]

    cam = point @ target2cam 
    #print (cam)
    #'''
    
    
    '''
    aru3 = get_camera_pose(
        image,
        {3: ((0, 0), 0.03)},
        K,
        dist,
        method=cv.SOLVEPNP_IPPE_SQUARE,
    )
    print(cam - aru3)
    #'''


    
    return target2cam



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
        cv.namedWindow('Camera image', cv.WINDOW_NORMAL)
        cv.imshow('Camera image', img)
        #cv2.waitKey(0) ## no waiting
    else:
        print("The image was not captured.")
 
    # Close communication with the camera before finish.
    camera.close()
    return img



if __name__ == "__main__":
    
    image = cv.imread("aruco.jpg")
    #image = scan_picture()
    target_to_cam(0,image)

