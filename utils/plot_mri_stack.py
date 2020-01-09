"""
Plot MRI slices stacked vertically
"""

import os
import matplotlib.pyplot as plt
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib._png import read_png
import scipy.misc
from matplotlib.collections import PolyCollection
from matplotlib import colors as mcolors
import cv2

data_folder = 'D:/Repos/MI-DQA/MRI_slices/'
image_files = os.listdir(data_folder)
import numpy
from matplotlib import pyplot
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

image = cv2.imread(data_folder+image_files[0], 0)
# Creat mesh.

xx, yy = numpy.mgrid[0:image.shape[0], 0:image.shape[1]]

# Create flat surface.
Z = image
print(Z)
# Plot
fig = pyplot.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(xx, yy, Z, rstride=1, cstride=1, linewidth=0, cmap=image)

ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

plt.show()