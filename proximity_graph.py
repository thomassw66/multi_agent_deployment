import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# takes a one dimentional array of points [ [x0, y0], [x1, y1], ...]
# and computes a square distance matrix to each point
#
def make_proximity_graph(points):
	x = points[:, 0]
	y = points[:, 1]
	dx = x[..., np.newaxis] - x[np.newaxis, ...]
	dy = y[..., np.newaxis] - y[np.newaxis, ...]
	return np.sqrt ( np.square(dx) + np.square(dy) );

# given a distance matrix return 1 if the distance is <= specified radius
#
def r_disk_graph(prox_graph, r):
    return np.less(prox_graph, r).astype(dtype=int) # this feels more sciency

# return a list of points neighboring point i
def get_neighborhood(i, points, adjacentcy_matrix):
    a = []
    for j in range(len(adjacentcy_matrix[i])):
        if i != j and adjacentcy_matrix[i][j]:
            a.append(points[j])
    return np.array(a)

# we call this to calculate an agents new direction
# all we should need is the points and its actual neighbors
# points [x, y]
def calculate_gradient(point, neighbors):
    return potential_field_gradient(point, neighbors) + 0.5 * ellipsoid_gradient(point, 1.0, 3.0)

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


############## COST FUNCTION GRADIENTS #############
# Ciruclar cost function gradient
def circle_gradient(point):
    return np.array([2*point[0], 2*point[1]], dtype='float64')

def ellipsoid_gradient(point, a, b):
    return np.array([2 * point[0] / a**2 , 2 * point[1] / b ** 2], dtype='float64')

def potential_field_gradient(point, neighbors):
    sum = np.array([0, 0], dtype='float64')
    for i in range(len(neighbors)):
        d = neighbors[i] - point
        distance = np.linalg.norm(d)
        sum += (distance ** -2 + 1/3 * distance ** -3) * normalize(d)
    return sum

# points = np.array([[1,1], [2,2], [3,3]])
# pg = make_proximity_graph(points)
# rd = r_disk_graph(pg, 1.5)
# print rd
# for i in range(len(points)):
#   print i
#    print get_neighborhood(i, points, rd)

fig = plt.figure()

plt.xlim(-5, 5)
plt.ylim(-5, 5)

"""
# Set up a contour plot of some function
x = np.linspace(-5, 5, 20);
y = np.linspace(-5, 5, 20);
[X, Y] = np.meshgrid(x , y);
Z = np.sqrt( np.square(X) + np.square(Y) )
plt.contour(X, Y, Z)
"""
# some random animations
# TODO: replace this with waypoint generation
TIME_STEPS = 50
STEP_SIZE = 0.1
MAX_RADIUS = 2.0

points = np.random.random([10, 2]) * 3 - 1.5
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
        # normalize gradient
        x[i] = x[i] - min(STEP_SIZE, size) * grad  # add the negative gradient to the point
    y.append(x);
y = np.array(y)

# graph initially empty
graph, = plt.plot([], [], 'o')

def animate(i):
    graph.set_data(y[i,:,0], y[i,:,1])
    return graph

ani = FuncAnimation(fig, animate, frames=TIME_STEPS, interval=2000/TIME_STEPS)
plt.show()

