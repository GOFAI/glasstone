import matplotlib.pyplot as plt
from matplotlib import cm, colors, colorbar
import numpy as np
from glasstone.fallout import WSEG10
from scipy.stats import lognorm

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

x = np.arange(-1, 10, 0.05)
y = np.arange(-1, 10, 0.05)
X, Y = np.meshgrid(x, y)
# use WSEG10's native units
w = WSEG10(gzx, gzy, yld, ff, wind_speed, wind_direction, wind_shear, dunits='mi', wunits='mph', yunits='MT', shearunits='mph/kilofoot')

def fatality_fraction(x, y):
    erd = w.dose(x, y, dunits='mi', doseunits='Roentgen')
    if erd > 2000.0:
        return 1.01
    else:
        return lognorm.cdf(erd, 0.42, scale=450)
    
deaths = np.vectorize(fatality_fraction)

Z = deaths(X, Y)

fig = plt.figure()
ax1 = fig.add_axes([0.1, 0.1, 0.7, 0.8])
ax2 = fig.add_axes([0.85, 0.1, 0.05, 0.75])
CS = ax1.contour(X, Y, Z, [0.1, 0.5, 0.75, 0.95, 1.0], colors=('b', 'g', 'c', 'y', 'r'), linewidths=2)
cmap = colors.ListedColormap(['b', 'g', 'c', 'y'])
cmap.set_over('r')
cmap.set_under('w')
norm = colors.BoundaryNorm([0.1, 0.5, 0.75, 0.95, 1.0], cmap.N)
cb = colorbar.ColorbarBase(ax2, cmap=cmap,
                           norm=norm,
                           boundaries=[0] + [0.1, 0.5, 0.75, 0.95, 1.0] + [1.1],
                           extend='both',
                           extendfrac='auto')
cb.set_label(r'Probability of Death')
ax1.grid(True)
ax1.set_title('WSEG-10 probability of death for unsheltered individual')
ax1.text(-0.7, 5.9, '$Wind: SW, 2.30303 mi/hr$\n$Shear: 0.23 mi/hr-kilofeet$\n$Yield: 10kT$\n$GZ:1,1$\n$FF: 1.0$\n$HOB: 0$\n$LD_{10}=263R$\n$LD_{50}=450R$\n$LD_{95}=900R$')
ax1.set_ylim([-0.5, 11])
ax1.set_ylabel('$st.$ $miles$')
ax1.set_xlabel('$st.$ $miles$')
plt.show()
