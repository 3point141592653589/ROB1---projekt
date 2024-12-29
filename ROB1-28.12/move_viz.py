import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# Nastavení limitů os (pro lepší vizualizaci)
#ax.set_xlim([0, 200])
#ax.set_ylim([-100, 100])
#ax.set_zlim([0, 200])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
#'''




def plot_transform(_,T,color='k', scale=1.0):
    """
    Vykreslí bod reprezentovaný transformační maticí T v 3D grafu.
    ax: Objekt 3D grafu
    T: Transformační matice 4x4
    scale: Velikost směrových os
    """
    # Extrakce polohy
    origin = T[:3, 3] * scale

    # Extrakce směrových os (x, y, z) z rotační části matice
    x_axis = T[:3, 0]
    y_axis = T[:3, 1]
    z_axis = T[:3, 2]

    # Vykreslení směrových os
    ax.quiver(*origin, *x_axis, color='r', label='X' if 'X' not in ax.get_legend_handles_labels()[1] else "")
    ax.quiver(*origin, *y_axis, color='g', label='Y' if 'Y' not in ax.get_legend_handles_labels()[1] else "")
    ax.quiver(*origin, *z_axis, color='b', label='Z' if 'Z' not in ax.get_legend_handles_labels()[1] else "")

    # Vykreslení bodu
    ax.scatter(*origin, color=color)  # Bod ve středu transformace
    fig.show()
    plt.pause(1)
    return None



if __name__=="__main__":


    # Příklad transformační matice: rotace + translace
    T1 = np.array([
        [1, 0, 0, 2],  # Rotace + posun (x=2)
        [0, 1, 0, 3],  # Rotace + posun (y=3)
        [0, 0, 1, 4],  # Rotace + posun (z=4)
        [0, 0, 0, 1]   # Homogenní souřadnice
    ])
    plot_transform(None,T1)
    #fig.show()
    #plt.pause(1)
    #plt.show(block=True)

    T2 = np.array([
        [0, -1, 0, 5],  # Rotace + posun (x=5)
        [1,  0, 0, 1],  # Rotace + posun (y=1)
        [0,  0, 1, 2],  # Rotace + posun (z=2)
        [0,  0, 0, 1]
    ])

    plot_transform(None,T2)
    #fig.show()
    #plt.pause(0)
    plt.show()



