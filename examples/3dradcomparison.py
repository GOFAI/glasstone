import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from glasstone.radiation import glasstone_summary, soviet_summary

zmax = 125000
zmin = 20

xmin = 1.0
xmax = 10.0

ymin = 500
ymax = 1000

yields = np.arange(xmin, xmax, 0.1)
slant_ranges = np.arange(ymin, ymax, 5)
X, Y = np.meshgrid(yields, slant_ranges)

def sovietsummarygraph(y, sr, scenario='summer'):
    try:
        dose = soviet_summary(y, sr, scenario=scenario)
    except ValueOutsideGraphError:
        if sr > 1400:
            dose = zmin
        else:
            dose = zmax
    if dose > zmax:
        return zmax
    elif dose < zmin:
        return zmin
    else:
        return dose

def glasstoneradgraph(y, sr, offense=True):
    dose = glasstone_summary(y, sr, 100.0, offense=offense)
    if dose > zmax:
        dose = zmax
    elif dose < zmin:
        dose = zmin
    else:
        return dose
    
    
ssg = np.vectorize(sovietsummarygraph)
grg = np.vectorize(glasstoneradgraph)

soviet = ssg(X, Y)
soviet2 = ssg(X, Y, scenario='mountain')
soviet3 = ssg(X, Y, scenario='winter')
american = grg(X, Y, offense=False)
american2 = grg(X, Y, offense=True)

# plt.style.use('dark_background')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_wireframe(X, Y, american, rstride=4, cstride=4, color='blue', label='U.S. (defense)')
ax.plot_wireframe(X, Y, soviet2, rstride=4, cstride=4, color='red', label='Soviet (mountain)')
ax.plot_wireframe(X, Y, soviet, rstride=4, cstride=4, color='deeppink', label='Soviet (summer)')
ax.plot_wireframe(X, Y, american2, rstride=4, cstride=4, color='dodgerblue', label='U.S. (offense)')
ax.plot_wireframe(X, Y, soviet3, rstride=4, cstride=4, color='magenta', label='Soviet (winter)')



ax.set_title('penetrating radiation from low-yield fission airbursts')
ax.legend(loc=2, fontsize='small', bbox_to_anchor=(0.0, 0.9))

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymax, ymin)
ax.set_zlim(zmin, zmax)
ax.set_xlabel('yield ($kT$)', fontsize='small')
ax.set_ylabel('slant range ($m$)', fontsize='small')
ax.set_zlabel('summary dose ($rads$)', fontsize='small', rotation='vertical')
ax.set_xticks([1.0, 5.0, 10.0])
ax.set_zticks([10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000])
ax.set_xticklabels(map(str, [1.0, 5.0, 10.0]), size='small')
ax.set_yticklabels(map(str, [x * 100 + 400 for x in xrange(1, 7)]), size='small')
ax.set_zticklabels(map(str, [x * 10000 for x in xrange(1, 13)]), size='small')

plt.show()
