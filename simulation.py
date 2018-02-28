import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import gradients as grads
import contours
from util import *
from scipy.spatial import Voronoi, voronoi_plot_2d


# we call this to calculate an agents new direction
# all we should need is the points and its actual neighbors
# points [x, y]
def calculate_gradient(point, neighbors):
    return 4 * grads.potential_field(point, neighbors) + \
            1.0 * grads.ellipsoid(point,0.7,0.7, center=(4,3)) + \
            1.0 * grads.ellipsoid(point,0.7,0.7, center=(-4,3))

# GENERATE DATA

TIME_STEPS = 150
STEP_SIZE = 0.05
MAX_RADIUS = 1

points = make_random_points(10, 2, center=(-3,-3)) # np.random.random([10, 2]) * 10 - 5
y = [points]
v = [Voronoi(points)]
for i in range(1, TIME_STEPS):
    x = y[i-1].copy()
    # ***** CALCULATE NEW POSITIONS HERE *****
    prox = make_proximity_graph(x)
    r_disk = r_disk_graph(prox, MAX_RADIUS)
    # calculate cost function gradient
    for i in range(len(x)):
        # update x[i]
        n = get_neighborhood(i, x, r_disk)
        grad = calculate_gradient(x[i], n)
        size = np.linalg.norm(grad)
        grad = normalize(grad)
        alpha = min(STEP_SIZE * size, STEP_SIZE)
        # check if intersection with an obstacle ??
        x[i] = x[i] - alpha * grad
    y.append(x)
    v.append(Voronoi(x))
y = np.array(y)


# ************* SHOW TRAJECTORIES ***********************************
fig, ax = plt.subplots()

ax.set_xlim((-5, 5))
ax.set_ylim((-5, 5))

# graph initially empty
# graph, = plt.plot([], [], 'o')

# voronoi
#voronoi, = ax.plot([], [], 'ro')
voronoi_lin_segs = []
for i in range(35):
    vor_lin_seg, = ax.plot([], [], 'k-', lw=1)
    voronoi_lin_segs.append(vor_lin_seg)

lines = []
for i in range(len(y[0])):
    line, = ax.plot([], [], 'b--')
    lines.append(line)

point, = ax.plot([],[], 'bo')

# draw obstacle


X,Y = np.meshgrid(np.linspace(-5,5, 100), np.linspace(-5, 5, 100))
Z = 1.0 * contours.ellipsoid(X, Y, 0.2, 0.2, center=(4,3)) + 1.0 * contours.ellipsoid(X, Y, 0.2, 0.2, center=(-4,3))
plt.contourf(X, Y, -Z, 50, cmap='gray')

def animate(i):
    # redraw agent trajectories (the blue path line)
    for j in range(len(y[i])):
            lines[j].set_data(y[:i, j, 0], y[:i, j, 1])
    # place a blue dot at each agents location
    point.set_data(y[i,:,0], y[i,:,1])

    # draw voronoi partitions
    vor = v[i]
    important_length = min(len(vor.ridge_vertices), len(voronoi_lin_segs))
    for i in range(important_length):
        simplex = np.asarray(vor.ridge_vertices[i])
        pointidx = vor.ridge_points[i] # is this always the same length?
        if np.all(simplex >= 0):
            voronoi_lin_segs[i].set_data(vor.vertices[simplex, 0], vor.vertices[simplex, 1])
        else:
            center = points.mean(axis=0)
            e = simplex[simplex >= 0][0] # finite end voronoi vertex
            # i = simplex[simplex >= 0][0] # finite end Voronoi vertex
            t = points[pointidx[1]] - points[pointidx[0]]  # tangent
            t = t / np.linalg.norm(t)
            n = np.array([-t[1], t[0]]) # normal
            midpoint = points[pointidx].mean(axis=0)
            # the far point is going to be along this line intersecting with the boundary of the environment
            far_point = vor.vertices[e] + np.sign(np.dot(midpoint - center, n)) * n * 10
            voronoi_lin_segs[i].set_data([vor.vertices[e,0], far_point[0]],
                  [vor.vertices[e,1], far_point[1]])


    for i in range(important_length, len(voronoi_lin_segs)):
        voronoi_lin_segs[i].set_data([], [])


    return point, voronoi_lin_segs

ani = FuncAnimation(fig, animate, frames=TIME_STEPS, interval=2000.0/TIME_STEPS)
plt.show()

