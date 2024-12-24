import cv2
import numpy as np
#: Our Basler camera interface
from kalibrace.basler_camera import BaslerCamera
from pathlib import Path
from ctu_crs import CRS97

def save_image_conf(robot, camera=None, pos_goal=None, id=None, dir="handeye"):
    if not id:
        id = len(list(Path(dir).glob("*"))) // 3

    if pos_goal is not None:
        move_ik(robot, pos_goal)

    not_camera = not camera
    if not_camera:
        camera = init_camera()

    img = camera.grab_image()


    if (img is not None) and (img.size > 0):
        cv2.namedWindow('Camera image', cv2.WINDOW_NORMAL)
        cv2.imshow('Camera image', img)
        cv2.waitKey(0)
    else:
        print("The image was not captured.")

    q_rad = robot.get_q( )
    pos = robot.fk(q_rad)
    q = robot.get_q()
    if pos_goal is None:
        pos_goal = pos
    cv2.imwrite(Path(dir) / Path(f"image_{id}.png"), img)
    np.save(Path(dir) / Path(f"pos_goal_{id}"), pos_goal)
    np.save(Path(dir) / Path(f"pos_{id}"), pos)
    np.save(Path(dir) / Path(f"joints_{id}"), q)
    

    
    if not_camera:
        camera.close()

def move_ik(robot, pos_goal):
    q_rad = robot.ik(pos_goal)
    q_radlim = [q for q in q_rad if robot.in_limits(q)]
    if not q_radlim:
        print("pose not reachable")
        return

    robot_q = robot.get_q()
    q = min(q_radlim, key=lambda x: np.linalg.norm(robot_q - x))

    robot.move_to_q (q)
    robot.wait_for_motion_stop ( )

def init_camera():
    camera: BaslerCamera = BaslerCamera()


    camera.connect_by_name("camera-crs97")

    camera.open()
    camera.set_parameters()
    camera.start()
    return camera

def init_robot():
    robot = CRS97()
    robot.reset_motors()
    robot.initialize(home=False)
    return robot
