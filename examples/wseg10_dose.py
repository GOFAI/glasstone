import matplotlib.pyplot as plt
from matplotlib import cm, colors, colorbar
import numpy as np
from glasstone.fallout import WSEG10

# ground zero x & y locations (st. mi)
gzx = 1
gzy = 1
# yield in megatons
yld = 0.01
# fission fraction
ff = 1.0
# wind speed (mph)
wind_speed = 1.151515 * 2.0
# wind direction (in degrees with wind from north = 0)
wind_direction = 225
# wind shear (change in mph per kilofoot change in altitude)
wind_shear = 0.23

x = np.arange(-1, 10, 0.1)
y = np.arange(-1, 10, 0.1)
X, Y = np.meshgrid(x, y)
# use WSEG10's native units
w = WSEG10(gzx, gzy, yld, ff, wind_speed, wind_direction, wind_shear, dunits='mi', wunits='mph', yunits='MT', shearunits='mph/kilofoot')

dose = np.vectorize(w.dose)

Z = dose(X, Y, dunits='mi', doseunits='Roentgen')

fig = plt.figure()
ax1 = fig.add_axes([0.1, 0.1, 0.7, 0.8])
ax2 = fig.add_axes([0.85, 0.1, 0.05, 0.75])
CS = ax1.contour(X, Y, Z, [500, 1000, 3000, 10000, 20000], colors=('b', 'g', 'c', 'y', 'r'), linewidths=2)
cmap = colors.ListedColormap(['b', 'g', 'c', 'y'])
cmap.set_over('r')
cmap.set_under('w')
norm = colors.BoundaryNorm([500, 1000, 3000, 10000, 20000], cmap.N)
cb = colorbar.ColorbarBase(ax2, cmap=cmap,
                           norm=norm,
                           boundaries=[0] + [500, 1000, 3000, 10000, 20000] + [30000],
                           extend='both',
                           extendfrac='auto')
cb.set_label(r'Equivalent Residual Dose $(Roentgens)$')
ax1.grid(True)
ax1.set_title('WSEG-10 30-day total dose contours for 10kT burst')
ax1.text(-0.5, 7.5, '$Wind: SW, 2.30303 mi/hr$\n$Shear: 0.23 mi/hr-kilofeet$\n$Yield: 10kT$\n$GZ:1,1$\n$FF: 1.0$\n$HOB: 0$')
ax1.set_ylim([-0.5, 11])
ax1.set_ylabel('$st.$ $miles$')
ax1.set_xlabel('$st.$ $miles$')
plt.show()
