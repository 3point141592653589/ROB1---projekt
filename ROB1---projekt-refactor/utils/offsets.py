import numpy as np

def patch_target2base(targ_to_base):
    angles = [0,0,0]
    angles[0] = -np.atan2(targ_to_base[2,1],targ_to_base[2,2])
    angles[1] = -np.atan2(-targ_to_base[2,0],np.sqrt(targ_to_base[2,1]**2+targ_to_base[2,2]**2))
    angles[2] = np.atan2(targ_to_base[1,0],targ_to_base[0,0])
    #'''

    #'''
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
    #angles[:2] = [0,0]

    cosx = np.cos(angles[0])
    sinx = np.sin(angles[0])
    cosy = np.cos(angles[1])
    siny = np.sin(angles[1])
    cosz = np.cos(angles[2])
    sinz = np.sin(angles[2])


    rotx = np.eye(3)*cosx
    rotx[0,0] = 1
    rotx[1,2] = -sinx
    rotx[2,1] = sinx

    roty = np.eye(3)*cosy
    roty[1,1] = 1
    roty[0,2] = siny
    roty[2,0] = -siny

    rotz = np.eye(3)*cosz
    rotz[2,2] = -1
    rotz[0,1] = -sinz
    rotz[1,0] = sinz

    rot = rotz @ roty @ rotx
    targ_to_base[:3,:3] = rot
    return targ_to_base
