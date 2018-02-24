import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import gradients as grads
from util import *
from scipy.spatial import Voronoi, voronoi_plot_2d


# we call this to calculate an agents new direction
# all we should need is the points and its actual neighbors
# points [x, y]
def calculate_gradient(point, neighbors):
    return 1.5 * grads.potential_field(point, neighbors) + \
            2.0 * grads.ellipsoid(point,1,0.7, center=(2,2))

# some random animations
# TODO: replace this with waypoint generation
TIME_STEPS = 150
STEP_SIZE = 0.1
MAX_RADIUS = 1

points = make_random_points(10, 1, center=(-4,-4)) # np.random.random([10, 2]) * 10 - 5
y = [points]
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
        # normalize gradient
        # check if intersection with an obstacle ??

        x[i] = x[i] - alpha * grad  # add the negative gradient to the point
    y.append(x);
y = np.array(y)


fig, ax = plt.subplots()

ax.set_xlim((-5, 5))
ax.set_ylim((-5, 5))

# graph initially empty
# graph, = plt.plot([], [], 'o')


lines = []
for i in range(len(y[0])):
    line, = ax.plot([], [], 'b', lw=2)
    lines.append(line)
# line, = ax.plot([], [], 'b', lw=2)
point, = ax.plot([],[], 'bo')

vor = Voronoi(y[0])
a = voronoi_plot_2d(vor)
print a

def animate(i):
    for j in range(len(y[i])):
            lines[j].set_data(y[:i, j, 0], y[:i, j, 1])
    # line.set_data(y[:i,:,0], y[:i,:,1])
    point.set_data(y[i,:,0], y[i,:,1])
    # graph.set_data(y[i,:,0], y[i,:,1])
    return point, lines

ani = FuncAnimation(fig, animate, frames=TIME_STEPS, interval=2000.0/TIME_STEPS)
plt.show()

