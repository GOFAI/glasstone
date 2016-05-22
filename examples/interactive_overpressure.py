# Interactive peak static overpressure plot using glasstone and matplotlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Cursor
from glasstone.overpressure import brode_overpressure

# Overpressure function from pp. 60-71 of H.L. Brode, _Airblast From Nuclear
# Bursts-Analytic Approximations_ (Pacific-Sierra Research Corporation, 1986).

plt.style.use('dark_background')
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)
t = np.arange(0.01, 25, 0.02)
y0 = 16
h0 = 2
s = brode_overpressure(y0, t, h0, dunits='kilofeet', opunits='psi')
l, = plt.plot(t, s, lw=2, color='blue')
plt.axis([0.01, 10, 0, 50])
plt.grid()

ax.set_title('interactive peak static overpressure calculator')

ax.set_xlabel('distance ($kilofeet$)')
ax.set_ylabel('peak static overpressure ($psi$)')

axyield = plt.axes([0.25, 0.05, 0.65, 0.03])
axheight = plt.axes([0.25, 0.01, 0.65, 0.03])

syield = Slider(axyield, 'Yield ($kT$):', 1.0, 1000.0, valinit=y0)
sheight = Slider(axheight, 'Burst height ($kilofeet$):', 0.0, 10.0, valinit=h0)

def update(val):
    bomb_yield = syield.val
    burst_height = sheight.val
    l.set_ydata(brode_overpressure(bomb_yield, t, burst_height, dunits='kilofeet', opunits='psi'))
    fig.canvas.draw_idle()
syield.on_changed(update)
sheight.on_changed(update)

cursor = Cursor(ax, useblit=False, color='red', linewidth=2 )

plt.show()
