import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d

plt.style.use('ggplot')
plt.rcParams["axes.facecolor"] = 'white'
fig = plt.figure()
ax = fig.gca(projection='3d')

# Make the grid
x, y, z = np.meshgrid(np.linspace(-1,1,3),
                      np.linspace(-1,1,3),
                      np.linspace(-1,1,3))

# Make the direction data for the arrows
u = 0.1*np.ones((3,3))
v = 0.1*np.ones((3,3))
w = 0.1*np.ones((3,3))
ax.set(xlabel='x',ylabel='y',zlabel='z')
ax.quiver(x, y, z, u, v, w)
ax.plot(x.ravel(),y.ravel(),z.ravel(),'go')
plt.savefig('vector_3d.jpg',dpi=120)
plt.show()