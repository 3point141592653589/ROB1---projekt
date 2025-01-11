import numpy as np


def rotationMatrixToEulerAngles(R):
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = np.atan2(R[2, 1], R[2, 2])
        y = np.atan2(-R[2, 0], sy)
        z = np.atan2(R[1, 0], R[0, 0])
    else:
        x = np.atan2(-R[1, 2], R[1, 1])
        y = np.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def patch_target2base(targ_to_base):
    angles = rotationMatrixToEulerAngles(targ_to_base)

    for i in range(2):
        if angles[i] > np.radians(12):
            angles[i] = np.radians(15)
        elif angles[i] < -np.radians(12):
            angles[i] = -np.radians(15)
        elif angles[i] > np.radians(5.5):
            angles[i] = np.radians(7)
        elif angles[i] < -np.radians(5.5):
            angles[i] = -np.radians(7)
        else:
            angles[i] = 0
    #'''

    cosx = np.cos(angles[0])
    sinx = np.sin(angles[0])
    cosy = np.cos(angles[1])
    siny = np.sin(angles[1])
    cosz = np.cos(angles[2])
    sinz = np.sin(angles[2])

    rotx = np.eye(3) * cosx
    rotx[0, 0] = 1
    rotx[1, 2] = -sinx
    rotx[2, 1] = sinx

    roty = np.eye(3) * cosy
    roty[1, 1] = 1
    roty[0, 2] = siny
    roty[2, 0] = -siny

    rotz = np.eye(3) * cosz
    rotz[2, 2] = 1
    rotz[0, 1] = -sinz
    rotz[1, 0] = sinz

    rot = rotz @ roty @ rotx
    targ_to_base[:3, :3] = rot
    return targ_to_base
