import numpy as np
import matplotlib.pyplot as plt

def plot_trajectory_animation(ax, x_t):
    lines = []
    for i in range(len(x_t[-1])):
        line, = ax.plot([], [], 'b--')
        lines.append(line)
    point, = ax.plot([],[], 'bo') 
    def animate(i):
        for j in range(len(x_t[i])):
            lines[j].set_data(x_t[:i, j, 0], x_t[:i, j, 1])
        point.set_data(x_t[i, :, 0], x_t[i, :, 1])
    return animate


