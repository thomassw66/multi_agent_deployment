#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import gradients as grads
import contours
from util import *
from scipy.spatial import Voronoi, voronoi_plot_2d
from plot_trajectory import *

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

X,Y = np.meshgrid(np.linspace(X_LIM[0],X_LIM[1], 100), np.linspace(Y_LIM[0], Y_LIM[1], 100))
Z = importance_function(X, Y)
plt.contourf(X, Y, -Z, 20, cmap='pink')

ax.set_xlim(X_LIM)
ax.set_ylim(Y_LIM)

animate = plot_trajectory_animation(ax, y[:,:-5,:], plot_vor=True)
ani = FuncAnimation(fig, animate, frames=TIME_STEPS, interval=2000.0/TIME_STEPS)
plt.show()

