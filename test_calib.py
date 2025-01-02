import cv2 as cv
import numpy as np
from ctu_crs import CRS97


def main():
    K = np.load("./cam_calib/cam_params/K.npy")
    dist = np.load("./cam_calib/cam_params/dist.npy")
    cam2base = np.load("./handeye_output_refined/cam2base.npy")
    dh_offset = np.load("./handeye_output_refined/dh_offset.npy")
    q = np.load("./handeye_data/joints_350.npy")
    img = cv.imread("./handeye_data/image_350.png")
    robot = CRS97(None)
    robot.dh_offset = dh_offset
    grip2base = robot.fk(q)

    grip = grip2base[:3, 3]
    base2cam = np.linalg.inv(cam2base)
    rvec, _ = cv.Rodrigues(base2cam[:3, :3])
    t = base2cam[:3, 3]
    point, *_ = cv.projectPoints(grip, rvec, t, K, dist)

    cv.drawMarker(
        img,
        point.astype("uint32").squeeze()[::-1],
        (0, 0, 255),
        markerSize=50,
        thickness=4,
    )
    cv.imshow("chuj", img)
    cv.waitKey(0)


if __name__ == "__main__":
    main()
