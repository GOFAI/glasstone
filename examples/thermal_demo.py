# Total thermal impulse plot using glasstone and matplotlib

import numpy as np
import matplotlib.pyplot as plt
from glasstone.thermal import soviet_ground_thermal, soviet_air_thermal
from glasstone.utilities import ValueOutsideGraphError

yld = 1000.0

slant_ranges = np.arange(3.0, 20.0, 0.25)

# these functions prevent cause low-visibility conditions to return 0.0 instead of
# ValueOutsideGraph errors:
def ground_thermal(sr, ivc):
    try:
        impulse = soviet_ground_thermal(yld, sr, 0.0, ivc, dunits='km')
    except ValueOutsideGraphError:
        impulse = 0.0
    return impulse

def air_thermal(sr, ivc):
    try:
        impulse = soviet_air_thermal(yld, sr, 70.0, ivc, dunits='km')
    except ValueOutsideGraphError:
        impulse = 0.0
    return impulse


ground = np.vectorize(ground_thermal)
air = np.vectorize(air_thermal)

ground_fog = ground(slant_ranges, 3)
ground_haze = ground(slant_ranges, 4)
ground_clear = ground(slant_ranges, 7)

air_fog = air(slant_ranges, 3)
air_haze = air(slant_ranges, 4)
air_clear = air(slant_ranges, 7)

plt.style.use('dark_background')

fig, (ax0, ax1) = plt.subplots(nrows=2)

fig.subplots_adjust(hspace=0.4)

lw = 2

ax0.set_xlim(3.0, 20.0)
ax0.set_ylim(0.2, 110)
ax1.set_xlim(3.0, 20.0)
ax1.set_ylim(0.4, 375)

ax0.plot(slant_ranges, ground_fog, label='light fog (IVC=3)', color='blue', linewidth=lw)
ax0.plot(slant_ranges, ground_haze, label='smog (IVC=4)', color='red', linewidth=lw)
ax0.plot(slant_ranges, ground_clear, label='clear air (IVC=7)', color='magenta', linewidth=lw)
ax0.set_xlabel('slant range ($km$)', fontsize='small')
ax0.set_ylabel('total thermal impulse ($Cal/cm^2$)', fontsize='small')
ax0.set_yscale('log')
ax0.grid(True)
ax0.set_title('total thermal impulse from 1MT groundburst')
ax1.plot(slant_ranges, air_fog, label='light fog (IVC=3)', color='blue', linewidth=lw)
ax1.plot(slant_ranges, air_haze, label='smog (IVC=4)', color='red', linewidth=lw)
ax1.plot(slant_ranges, air_clear, label='clear air (IVC=7)', color='magenta', linewidth=lw)
ax1.set_xlabel('slant range ($km$)', fontsize='small')
ax1.set_ylabel('total thermal impulse ($Cal/cm^2$)', fontsize='small')
ax1.set_yscale('log')
ax1.grid(True)
ax1.set_title('total thermal impulse from 1MT airburst')
ax0.legend()

plt.show()
