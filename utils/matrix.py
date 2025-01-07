import numpy as np


def axis_rot_matrix(th: float, axis: str):
    cos = np.cos(th)
    sin = np.sin(th)
    if axis == "x":
        return np.array([[1, 0, 0], [0, cos, -sin], [0, sin, cos]])
    elif axis == "y":
        return np.array([[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]])
    elif axis == "z":
        return np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])
    e = "invalid rotation axis, use only xyz"
    raise RuntimeError(e)


def euler2mat(theta_axis_list: list[tuple[float, str]]):
    ret = np.eye(4)
    for th, axis in theta_axis_list:
        ret = ret @ axis_rot_matrix(th, axis)
    return ret
