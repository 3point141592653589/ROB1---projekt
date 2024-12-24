import numpy as np
from util import save_image_conf
from ctu_crs import CRS97
robot = CRS97()




def rot_matrix(th, axis):

    cos = np.cos(th)
    sin = np.sin(th)
    if axis == 'x':
        return np.array([[1,0,0],[0,cos, -sin],[0,sin,cos]])
    elif axis == 'y':
        return np.array([[cos,0,-sin],[0,1, 0],[sin,0,cos]])
    elif axis == 'z':
        return np.array([[cos,-sin,0],[sin,cos,0],[0,0,1]])



def point_rots(angles):
    
    rot_y = rot_matrix(np.pi, "y")
    rots = [rot_y]
    for angle in angles:
        if angle != np.deg2rad(-30):
            rots.append(rot_matrix(angle, "y")@rot_y)
        rots.append(rot_matrix(angle, "z")@rot_y)
        r = rot_matrix(np.pi/4, "x")
        rots.append(rot_matrix(angle, "y") @ r @ rot_y)
        r = rot_matrix(-np.pi/4, "x")
        rots.append(rot_matrix(angle, "y") @ r @ rot_y)
    return rots

def point_save_rots_imgs(v, angles, robot, camera=None, dir="handeye_data"):
    rots = point_rots(angles)
    for rot in rots:
        pos = np.eye(4)
        pos[:3, :3] = rot
        pos[:3, 3] = v
        try:
            save_image_conf(robot, camera, pos, dir=dir)
        except Exception as e:
            print(e)
            print("orientation IK failed:")
            print(pos)



robot.reset_motors()
robot.initialize(home=False)
#robot.initialize()
#robot.soft_home ( )

v = np.array([0.35, -0.12, 0.180])
angles = np.deg2rad([-30, -15, 15, 30])

point_save_rots_imgs(v, angles, robot)

robot.release()