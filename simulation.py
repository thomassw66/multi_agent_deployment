import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import gradients as grads
import contours
from util import *
from scipy.spatial import Voronoi, voronoi_plot_2d

# General Parameters 
integration_resolution = 50
NUM_ADVERSARIES = 6
TIME_STEPS = 50
STEP_SIZE = 0.3
MAX_RADIUS = 2.9
L = 10
A = 0.5
B = 0.5
K = 500
R = 1

important_centers = [(-3, 0),(3,3) / np.sqrt(2), (4.5,4.5), (-4.5, 4.5), (0,0)]

NUM_AGENTS = 16
agent_center = (-3, -3)
agent_radius = 0.2

# Boundaries of our environment 
X_LIM = (-5, 5) 
Y_LIM = (-5, 5)

# we will define this to be a multi-modal gaussian centered at each of the centers in importance centers 
importance_function = lambda X, Y: sum( [ np.exp(-np.square(X-c[0])-np.square(Y-c[1])) for c in important_centers] )

# This is a helper for computing the gradient (grad H(P)) of our objective function 
dist_square = lambda X, Y, center: np.square(X - center[0]) + np.square(Y-center[1])

points = make_random_points(NUM_AGENTS, agent_radius, center=agent_center) 

"""
These are the location of our "spoofed aversaries" they are not intelligent and do not move 
"""
# TODO: make this less ugly 
points[NUM_AGENTS - NUM_ADVERSARIES: NUM_AGENTS] = [
    [-.5, .2], 
    [.0, 1.], 
    [.2, -.5],
    [.4, .4],
    [-.6, -0.6],
    [ 1.0, 0.0],
    ][:NUM_ADVERSARIES] + important_centers[1] if NUM_ADVERSARIES > 0 else np.zeros((0,2))

# we store points and voronoi diagrams at each time step 
y = [points]
v = [Voronoi(points)]

# run our algorithm for TIME_STEPS 
for t in range(1, TIME_STEPS):
    
    print "********** Time Step ", t, " / ", TIME_STEPS ," **************"
    x = y[t-1].copy()
    vor_partition = v[-1]

    # ***** CALCULATE NEW POSITIONS HERE *****
    for i in range(len(x)):
        if i < NUM_AGENTS - NUM_ADVERSARIES:
            # M = integr {Vi} f(|q - pi|) phi(q) dq
            mass = integrate_r_limited_voronoi(
                lambda X, Y: dist_square(X,Y, x[i]) * importance_function(X, Y), 
                x, i, x[i], MAX_RADIUS, vor_partition, integration_resolution
            )
            # Mx = integr {Vi} xi * f(|q - pi|) phi(q) dq
            mx = integrate_r_limited_voronoi(
                lambda X, Y: X * dist_square(X,Y,x[i]) * importance_function(X, Y),
                x, i, x[i], MAX_RADIUS, vor_partition, integration_resolution
            )
            # My = integr {Vi} yi * f(|q - pi|) phi(q) dq 
            my = integrate_r_limited_voronoi(
                lambda X, Y: Y * dist_square(X,Y, x[i]) * importance_function(X, Y), 
                x, i, x[i], MAX_RADIUS, vor_partition, integration_resolution
            )
            if mass > 0:
                mx /= mass
                my /= mass
            
            centroid = [mx, my]
            
            # moves in the direction of centriod
            grad = centroid - x[i]  
            size = np.linalg.norm(grad)
            grad = normalize(grad)
            alpha = min(STEP_SIZE * size, STEP_SIZE)
            # check if intersection with an obstacle ??
            x[i] = x[i] + alpha * grad
        else:
            # for this example the adversaries positions are static 
            x[i] = x[i]  
    
    # save our calculations to display later 
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
for i in range(NUM_AGENTS*4):
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
    vor = v[i]
    important_length = min(len(vor.ridge_vertices), len(voronoi_lin_segs))
    for i in range(important_length):
    	simplex = np.asarray(vor.ridge_vertices[i])
    	pointidx = vor.ridge_points[i] # is this always the same length?
     	if np.all(simplex >= 0):
    		voronoi_lin_segs[i].set_data(vor.vertices[simplex, 0], vor.vertices[simplex, 1])
    else:
	    voronoi_lin_segs[i].set_data([], [])

    for i in range(important_length, len(voronoi_lin_segs)):
        voronoi_lin_segs[i].set_data([], [])

    return point

ani = FuncAnimation(fig, animate, frames=TIME_STEPS, interval=2000.0/TIME_STEPS)
plt.show()

