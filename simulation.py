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
integration_resolution = 50
NUM_ADVERSARIES = 0
TIME_STEPS = 50
STEP_SIZE = 0.3
MAX_RADIUS = 1
L = 10
A = 0.5
B = 0.5
K = 500
R = 1
#important_center1 = (-3, 0.0)
#important_center2 = (3, 3.0) / np.sqrt(2)
important_centers = [ (-3, 0.0),(3,3) / np.sqrt(2),(0,-3)]

NUM_AGENTS = 15
agent_center = (-0.0, 0)
agent_radius = 0.2

X_LIM = (-5, 5)
Y_LIM = (-5, 5)

# smooth_ramp
SR = lambda x: x * (np.arctan(L * x) / np.pi + 0.5)

def ellipse(xc,yc,a,b,r):
	return lambda X, Y: a*(X - xc)**2 + b*(Y - yc)**2 - r**2

importance_function = lambda X, Y: sum([np.exp(-K * SR( ellipse(c[0],c[1],A,B,R)(X,Y) )) for c in important_centers])

dist_square = lambda X, Y, center: np.square(X - center[0]) + np.square(Y-center[1])

points = make_random_points(NUM_AGENTS, agent_radius, center=agent_center) # np.random.random([10, 2]) * 10 - 5
angles = np.linspace(np.pi, np.pi*3/2, NUM_ADVERSARIES)
points[NUM_AGENTS - NUM_ADVERSARIES: NUM_AGENTS] = [
    [-1.5, -0.8], 
    [.0, 1.], 
    [-.8, -1.5],
    [-0.6, -0.6],
    [-1.0, -0.2],
    [ 1.0, 0.0],
    ][:NUM_ADVERSARIES] + important_centers[1] if NUM_ADVERSARIES > 0 else np.zeros((0,2))

y = [points]
v = [Voronoi(points)]
for t in range(1, TIME_STEPS):
    print "********** Time Step ", t, " **************"
    x = y[t-1].copy()
    # ***** CALCULATE NEW POSITIONS HERE *****
    #prox = make_proximity_graph(x)
    #r_disk = r_disk_graph(prox, MAX_RADIUS)
    # calculate cost function gradient
    for i in range(len(x)):
        if i < NUM_AGENTS - NUM_ADVERSARIES: 
            mass = integrate_r_limited_voronoi(
                lambda X, Y: dist_square(X,Y, x[i]) * importance_function(X, Y), 
                x, i, x[i], MAX_RADIUS, v[-1], integration_resolution
            )
            mx = integrate_r_limited_voronoi(
                lambda X, Y: X * dist_square(X,Y,x[i]) * importance_function(X, Y),
                x, i, x[i], MAX_RADIUS, v[-1], integration_resolution
            )
            my = integrate_r_limited_voronoi(
                lambda X, Y: Y * dist_square(X,Y, x[i]) * importance_function(X, Y), 
                x, i, x[i], MAX_RADIUS, v[-1], integration_resolution
            )
            print "i: ", i, mass, mx, my
            if mass > 0:
                mx /= mass
                my /= mass
            
            grad = [mx, my] - x[i]  # moves in the direction of centriod
            size = np.linalg.norm(grad)
            # if size <= 1:
            grad = normalize(grad)
            alpha = min(STEP_SIZE * size, STEP_SIZE)
            # check if intersection with an obstacle ??
            x[i] = x[i] + alpha * grad
        else:
            x[i] = x[i] # stay where you are 
    y.append(x)
    v.append(Voronoi(x))
y = np.array(y)


# ************* SHOW TRAJECTORIES ***********************************
fig, ax = plt.subplots()

ax.set_xlim(X_LIM)
ax.set_ylim(Y_LIM)

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
a_point, = ax.plot([], [], 'ro')
# draw obstacle


X,Y = np.meshgrid(np.linspace(X_LIM[0],X_LIM[1], 100), np.linspace(Y_LIM[0], Y_LIM[1], 100))
Z = importance_function(X, Y)
plt.contourf(X, Y, -Z, 20, cmap='pink')

def animate(i):
    # redraw agent trajectories (the blue path line)
    for j in range(len(y[i])):
            lines[j].set_data(y[:i, j, 0], y[:i, j, 1])
    # place a blue dot at each agents location
    point.set_data(y[i,:NUM_AGENTS-NUM_ADVERSARIES,0], y[i,:NUM_AGENTS-NUM_ADVERSARIES,1])
    a_point.set_data(y[i,NUM_AGENTS-NUM_ADVERSARIES:,0], y[i, NUM_AGENTS-NUM_ADVERSARIES:,1])

    # draw voronoi partitions
    #vor = v[i]
    #important_length = min(len(vor.ridge_vertices), len(voronoi_lin_segs))
    #for i in range(important_length):
    #	simplex = np.asarray(vor.ridge_vertices[i])
    #	pointidx = vor.ridge_points[i] # is this always the same length?
    # 	if np.all(simplex >= 0):
    #		voronoi_lin_segs[i].set_data(vor.vertices[simplex, 0], vor.vertices[simplex, 1])
 	#else:
            #center = points.mean(axis=0)
            #e = simplex[simplex >= 0][0] # finite end voronoi vertex
            # i = simplex[simplex >= 0][0] # finite end Voronoi vertex
            #t = points[pointidx[1]] - points[pointidx[0]]  # tangent
            #t = t / np.linalg.norm(t)
            #n = np.array([-t[1], t[0]]) # normal
            #midpoint = points[pointidx].mean(axis=0)
            # the far point is going to be along this line intersecting with the boundary of the environment
            #far_point = vor.vertices[e] + np.sign(np.dot(vor.vertices[e] - center, n)) * n * 100
            #voronoi_lin_segs[i].set_data([vor.vertices[e,0], far_point[0]],
            #      [vor.vertices[e,1], far_point[1]])
	#	voronoi_lin_segs[i].set_data([], [])


    #for i in range(important_length, len(voronoi_lin_segs)):
    #3    voronoi_lin_segs[i].set_data([], [])


    return point

ani = FuncAnimation(fig, animate, frames=TIME_STEPS, interval=2000.0/TIME_STEPS)
plt.show()

