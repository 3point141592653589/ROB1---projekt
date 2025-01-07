import cv2 as cv
import numpy as np

img = cv.imread("./im1.bmp")
K = np.load("./cam_params/30mm/K.npy")
dist = np.load("./cam_params/30mm/dist.npy")
h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
print(K)
print(newcameramtx)
dst = cv.undistort(img, K, dist, None, newcameramtx)
cv.imshow("dist", img)
cv.waitKey(0)
cv.imshow("undist", dst)
cv.waitKey(0)
