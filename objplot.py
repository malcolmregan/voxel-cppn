import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def plotarray(array):
    filled=np.nonzero(array>0)
    x=filled[0]
    y=filled[1]
    z=filled[2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    plt.show()
