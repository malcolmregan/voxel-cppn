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
    
    axes = plt.gca()
    axes.set_xlim([0,32])
    axes.set_ylim([0,32])
    axes.set_zlim([0,32])

    ax.set_xlabel('input[0]')
    ax.set_ylabel('input[1]')
    ax.set_zlabel('input[2]')

    plt.show()

def plotsave(array, filename):
    filled=np.nonzero(array>0)
    x=filled[0]
    y=filled[1]
    z=filled[2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)

    axes = plt.gca()
    axes.set_xlim([0,32])
    axes.set_ylim([0,32])
    axes.set_zlim([0,32])

    ax.set_xlabel('input[0]')
    ax.set_ylabel('input[1]')
    ax.set_zlabel('input[2]')

    plt.savefig(filename)
    plt.close()
