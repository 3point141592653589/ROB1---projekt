import numpy as np
from ctu_crs import CRS97

from hardware_control import capture_grid

robot = CRS97()
robot.reset_motors()
robot.initialize(home=False)
# robot.soft_home ( )

rots = [
    [
        (np.pi, "y"),
        (np.pi / 2, "z"),
    ],
    [
        (np.pi, "y"),
        (np.pi / 2, "z"),
    ],
    [
        (np.pi, "y"),
        (np.pi / 2, "z"),
    ],
    [
        (5 / 6 * np.pi, "y"),
        (np.pi / 2, "z"),
    ],
    [
        (5 / 6 * np.pi, "y"),
        (np.pi / 2, "z"),
    ],
    [
        (5 / 6 * np.pi, "y"),
        (np.pi / 2, "z"),
    ],
    [
        (8 / 7 * np.pi, "y"),
        (np.pi * 5 / 8, "z"),
    ],
    [
        (8 / 7 * np.pi, "y"),
        (np.pi * 5 / 8, "z"),
    ],
    [
        (np.pi * 5 / 6, "x"),
        (-np.pi / 2, "z"),
    ],
    [
        (np.pi * 5 / 6, "x"),
        (-np.pi / 2, "z"),
    ],
    [
        (np.pi * 5 / 6, "x"),
        (-np.pi / 2, "z"),
    ],
]
corns1 = [
    np.array([0.38, -0.02, 0.2]),
    np.array([0.36, -0.05, 0.07]),
    np.array([0.375, -0.03, 0.14]),
    np.array([0.37, -0.03, 0.2]),
    np.array([0.36, -0.05, 0.07]),
    np.array([0.37, -0.04, 0.14]),
    np.array([0.425, -0.03, 0.2]),
    np.array([0.43, -0.03, 0.15]),
    np.array([0.4, -0.03, 0.17]),
    np.array([0.38, -0.05, 0.13]),
]
corns2 = [
    np.array([0.53, 0.26, 0.2]),
    np.array([0.54, 0.29, 0.07]),
    np.array([0.53, 0.27, 0.14]),
    np.array([0.46, 0.1, 0.2]),
    np.array([0.49, 0.14, 0.07]),
    np.array([0.48, 0.13, 0.14]),
    np.array([0.59, 0.26, 0.2]),
    np.array([0.595, 0.27, 0.15]),
    np.array([0.54, 0.21, 0.17]),
    np.array([0.54, 0.22, 0.13]),
]

for rot, corn1, corn2 in zip(rots, corns1, corns2):
    capture_grid(corn1, corn2, robot, (5, 10), euler_rot=rot)

# robot.release()
