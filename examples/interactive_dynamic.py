# Interactive peak static overpressure and dynamic pressure plot using glasstone and
# matplotlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Cursor
from glasstone.overpressure import DNA_static_overpressure, DNA_dynamic_pressure

# using Defense Nuclear Agency 1kT free airburst overpressure

static_overpressure = np.vectorize(DNA_static_overpressure)
dynamic_pressure = np.vectorize(DNA_dynamic_pressure)

plt.style.use('dark_background')
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)
t = np.arange(200, 10000, 20)
y0 = 16.0
h0 = 600.0
s1 = static_overpressure(y0, t, h0)
s2 = dynamic_pressure(y0, t, h0)
l1, = plt.plot(t, s1, lw=2, color='magenta', label='static overpressure')
l2, = plt.plot(t, s2, lw=2, color='blue', label='dynamic pressure')
plt.axis([200, 3000, 0, 3.5])
plt.grid()

ax.set_title('interactive peak static overpressure and dynamic pressure calculator')

ax.set_xlabel('distance ($m$)')
ax.set_ylabel('peak static overpressure / dynamic pressure ($kg/cm^2$)')
ax.legend()

axyield = plt.axes([0.25, 0.05, 0.65, 0.03])
axheight = plt.axes([0.25, 0.01, 0.65, 0.03])

syield = Slider(axyield, 'Yield ($kT$):', 1.0, 1000.0, valinit=y0)
sheight = Slider(axheight, 'Burst height ($m$):', 0.0, 5000.0, valinit=h0)

def update(val):
    bomb_yield = syield.val
    burst_height = sheight.val
    l1.set_ydata(static_overpressure(bomb_yield, t, burst_height))
    l2.set_ydata(dynamic_pressure(bomb_yield, t, burst_height))
    fig.canvas.draw_idle()

syield.on_changed(update)
sheight.on_changed(update)

cursor = Cursor(ax, useblit=False, color='red', linewidth=2 )

plt.show()
