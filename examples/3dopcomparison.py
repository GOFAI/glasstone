from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
from glasstone.overpressure import brode_overpressure, soviet_overpressure

# comparison of U.S. and Soviet models of peak static overpressure from a nuclear
# explosion
usoverp = np.vectorize(brode_overpressure)
suoverp = np.vectorize(soviet_overpressure)

burst_heights = np.arange(70, 200, 1)
ground_ranges = np.arange(70, 200, 1)
X, Y = np.meshgrid(burst_heights, ground_ranges)

soviet = suoverp(1.0, Y, X)
american = usoverp(1.0, Y, X)

plt.style.use('dark_background')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_wireframe(X, Y, american, rstride=7, cstride=7, color='blue', label='U.S. (ideal surface)')
ax.plot_wireframe(X, Y, soviet, rstride=7, cstride=7, color='red', label='Soviet (thermal precursor)')

ax.set_title('peak static overpressure for 1kT burst')
ax.legend(loc=0, fontsize='small', bbox_to_anchor=(1.0, 0.9))

ax.set_xlim(70, 200)
ax.set_ylim(70, 200)
ax.set_zlim(0, 23)
ax.set_xlabel('burst height ($m$)', fontsize='small')
ax.set_ylabel('ground range ($m$)', fontsize='small')
ax.set_zlabel('peak static overpressure ($kg/cm^2$)', fontsize='small')

plt.show()


