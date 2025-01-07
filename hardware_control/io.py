from pathlib import Path
import warnings

from ctu_crs.crs_robot import CRSRobot
import cv2
import numpy as np

#: Our Basler camera interface
from .camera import init_camera


def get_id(dir: str, fname: str, ftype: str):
    return len(list(Path(dir).glob(f"**/{fname}_*.{ftype}")))


def save_image_conf(robot: CRSRobot, camera=None, id=None, dir="handeye"):
    if not id:
        id = get_id(dir, "image", "png")

    not_camera = not camera
    if not_camera:
        camera = init_camera()

    img = camera.grab_image()

    if (img is not None) and (img.size > 0):
        cv2.namedWindow("Camera image", cv2.WINDOW_NORMAL)
        cv2.imshow("Camera image", img)
        cv2.waitKey(1)
    else:
        warnings.warn("Image was not captured")

    q_rad = robot.get_q()
    pos = robot.fk(q_rad)
    q = robot.get_q()
    cv2.imwrite(str(Path(dir) / Path(f"image_{id}.png")), img)
    np.save(Path(dir) / Path(f"pos_{id}"), pos)
    np.save(Path(dir) / Path(f"joints_{id}"), q)

    if not_camera:
        camera.close()
