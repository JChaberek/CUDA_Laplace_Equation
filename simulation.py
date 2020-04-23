import numpy as np
import matplotlib.pyplot as plt
import csv
from mpl_toolkits.mplot3d import Axes3D
import os, sys
size = 1024
center = size / 2

with open('.csv') as csvfile:
    radius = list(csv.reader(csvfile))
with open('Data.csv') as csvfile:
    data = list(csv.reader(csvfile))

R = int(radius[0][0])
arraydata = [data[i][0:size] for i in range(size)]
arraydata = np.concatenate([np.array(i) for i in arraydata])
arraydata.resize((size, size))
arraydata = np.array(arraydata, dtype=float)

fig, ax = plt.subplots()
plt.figure(0)
plt.imshow(arraydata, cmap='viridis', aspect='auto', interpolation='nearest')
plt.colorbar()
plt.title('electrostatic potential')
plt.figure(1)
x, y = np.mgrid[(center-R):(center+R), (center-R):(center+R)]
ax2 = plt.axes(projection="3d")
data = arraydata[int(center-R):int(center+R), int(center-R):int(center+R)]
ax2.plot_surface(x, y, data, cmap='viridis')
plt.title('electrostatic potential 3D')

plt.show()
