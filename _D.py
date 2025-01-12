import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# from ctu_crs import CRS93
# from ctu_crs import CRS97
from hardware_control.robot.crs_class_patch import CRS97Patch

from hardware_control.camera import init_camera
from hardware_control.operations import move_cubes
from utils.offsets import patch_target2base
from utils.tray import get_tray_dict
from vision.marker_pose import ArucoPoseSolver, get_marker_pose


def get_cube_data(tray_id):
    i = tray_id
    j = i + 1
    name = f"desky/positions_plate_0{i}-0{j}.csv"
    return np.loadtxt(name, delimiter=",", skiprows=1)


# robot = None
robot = CRS97Patch(tty_dev="/dev/mars")


robot.initialize(home=False)

if robot is not None:
    robot.gripper.control_position(600)
    robot.soft_home()
    q = robot.get_q()
    q[0] += np.pi / 8
    robot.move_to_q(q)
    robot.wait_for_motion_stop()

camera = init_camera()
image = camera.grab_image()


patch_t2b = False
cam_to_base = np.load("./calibration_data/handeye_output_refined/cam2base.npy")
K = np.load("./calibration_data/cam_params2/K.npy")
dist = np.load("./calibration_data/cam_params2/dist.npy")
if robot is not None:
    robot.dh_offset = np.load("./calibration_data/handeye_output_refined/dh_offset.npy")


def get_targ_to_base_function(target_id):
    t2c_solver = ArucoPoseSolver(get_tray_dict(target_id, target_id + 1), K, dist)

    def f(img):
        t2b = cam_to_base @ get_marker_pose(img, t2c_solver)
        return patch_target2base(t2b) if patch_t2b else t2b

    return f


if cam_to_base is None:
    print("No cam2base data")
    exit(1)


detector = cv.aruco.ArucoDetector(
    cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50),
)

corners, ids, rejected = detector.detectMarkers(image)
if ids is None:
    print("No markers detected")
    exit(1)
ids = np.array(ids)
ids = ids[:, 0]


if len(ids) != 4:
    print("wrong number of markers detected")
    exit(1)
sorted_ids = np.sort(ids)

while True:
    print(
        "Choose bloks source desk: 0:(IDs:",
        sorted_ids[0],
        ",",
        sorted_ids[1],
        "),1:(IDs:",
        sorted_ids[2],
        ",",
        sorted_ids[3],
        ")",
    )
    num_input = input()
    if num_input.isdigit():
        direction = int(num_input)
        break
    else:
        print("It's not a number")
if direction:
    direction = 1

t1id = sorted_ids[2 * direction]
t2id = sorted_ids[2 * (1 - direction)]

move_cubes(
    robot,
    camera,
    get_targ_to_base_function(t1id),
    get_targ_to_base_function(t2id),
    get_cube_data(t1id),
    get_cube_data(t2id),
)

if robot is not None:
    robot.soft_home()
    robot.release()
    robot.close()
else:
    plt.show()
