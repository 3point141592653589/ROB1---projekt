from pathlib import Path

import cv2
import numpy as np

#: Our Basler camera interface


def save_image_conf(robot, camera=None, pos_goal=None, id=None, dir="handeye"):
    if not id:
        id = len(list(Path(dir).glob("*"))) // 4

    if pos_goal is not None:
        move_ik(robot, pos_goal)

    not_camera = not camera
    if not_camera:
        camera = init_camera()

    img = camera.grab_image()

    if (img is not None) and (img.size > 0):
        cv2.namedWindow("Camera image", cv2.WINDOW_NORMAL)
        cv2.imshow("Camera image", img)
        cv2.waitKey(1)
    else:
        print("The image was not captured.")

    q_rad = robot.get_q()
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
