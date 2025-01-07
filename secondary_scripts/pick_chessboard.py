from pathlib import Path

import cv2 as cv
from tqdm import tqdm

imgs_path = "./chessboard30/"
picked_path = Path("./chessboard30picked")
chessboard_dims = (9, 6)
images = Path(imgs_path).glob("*")
picked_path.mkdir(exist_ok=True)


for fname in tqdm(list(images)):
    img = cv.imread(str(fname))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, chessboard_dims, None)

    # If found, add object points, image points (after refining them)
    if ret:
        # termination criteria
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv.cornerSubPix(gray, corners, (21, 21), (-1, -1), criteria)

        # Draw and display the corners
        cv.drawChessboardCorners(img, chessboard_dims, corners2, ret)
        cv.imshow("img", img)
        cv.waitKey(0)
        cv.destroyAllWindows()
        cajk = input("\ncajk?")
        if "y" in cajk:
            try:
                (picked_path / fname.name).symlink_to(fname)
            except FileExistsError:
                pass
