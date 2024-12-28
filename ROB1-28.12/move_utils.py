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

def move_grid(pose, corn1, corn2, robot, grid_shape=(4, 5), camera=None, dir="handeye_data/"):
    x1, y1 = corn1[:2]
    x2, y2 = corn2[:2]
    nx, ny = grid_shape

    for x in np.linspace(x1, x2, nx):
        for y in np.linspace(y1, y2, ny):
            pose[:2, 3] = np.array([x, y])
            try:
                save_image_conf(robot, camera, pose, dir=dir)
            except Exception as e:
                print(e)
                print("orientation IK failed:")
                print(pose)


if __name__ == "__main__":
    robot.reset_motors()
    robot.initialize(home=False)
    #robot.initialize()
    #robot.soft_home ( )

    '''
    corn1 = np.array([0.46, 0.22,0.235])
    corn2 = np.array([0.46-0.155, 0.22-0.28, 0.235])
    #'''
    '''
    corn1 = np.array([0.47, 0.26, 0.07])
    corn2 = np.array([0.47-0.1425, 0.26-0.36, 0.07])
    #'''
    '''
    corn1 = np.array([0.435, 0.19, 0.44])
    corn2 = np.array([0.435-0.12, 0.19-0.22, 0.44])
    #'''
    '''
    poz = np.eye(4)
    poz[:3, :3] = rot_matrix(np.pi, "y") @ rot_matrix(np.pi/4, "z")
    #'''



    #'''
    corn1 = np.array([0.45, -0.25,0.235])
    corn2 = np.array([0.46-0.155, -(0.22-0.28), 0.235])
    #'''
    '''
    corn1 = np.array([0.49, -0.3, 0.07])
    corn2 = np.array([0.47-0.13, -(0.26-0.34), 0.07])
    #'''
    '''
    corn1 = np.array([0.435-0.12, -(0.19-0.21), 0.44])
    corn2 = np.array([0.425, -0.21, 0.44])
    #'''
    
    poz = np.eye(4)
    poz[:3, :3] = rot_matrix(np.pi, "y") @ rot_matrix(-np.pi/4, "z")



    poz[2, 3] = corn1[2]
    

    move_grid(poz, corn1, corn2, robot)

    robot.release()