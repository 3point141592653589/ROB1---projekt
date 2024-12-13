import cv2 as cv
import numpy as np


def main():
    K = np.load("./cam_calib/cam_params/K.npy")
    dist = np.load("./cam_calib/cam_params/dist.npy")
    cam2base = np.load("./handeye_output/cam2base.npy")
    base2grip = np.load("./handeye_data/pos_39.npy")
    img = cv.imread("./handeye_data/image_39.png")

    grip = base2grip[:3, 3]
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
