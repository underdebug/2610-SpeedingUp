
# control_points = cpu('control_points', Size=int(gdb.parse_and_eval('cp_u')) * int(gdb.parse_and_eval('cp_v')))
# print(control_points)

import matplotlib
matplotlib.use('QtAgg')

import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = np.linspace(-2, 2, 30)
y = np.linspace(-2, 2, 30)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) * np.cos(Y)

ax.plot_surface(X, Y, Z)

plt.show()
