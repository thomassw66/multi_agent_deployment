import numpy as np
import matplotlib.pyplot as plt

def plot_trajectory_animation(ax, x_t, plot_vor=False):
    lines = []
    for i in range(len(x_t[-1])):
        line, = ax.plot([], [], 'b--')
        lines.append(line)
    point, = ax.plot([],[], 'bo') 
    if plot_vor:
        # plot voronoi regions for each t
        from scipy.spatial import Voronoi, voronoi_plot_2d
        v = [Voronoi(x) for x in x_t]
        el = max([len(vi.ridge_vertices) for vi in v])
        v_lines = [ax.plot([], [], 'k-', lw=1)[0] for i in range(el)]

    def animate(i):
        for j in range(len(x_t[i])):
            lines[j].set_data(x_t[:i, j, 0], x_t[:i, j, 1])
        point.set_data(x_t[i, :, 0], x_t[i, :, 1])
        if plot_vor:
            print len(v_lines)
            for j in range(len(v_lines)):
                if j >= len(v[i].ridge_vertices):
                    continue 
                simplex = np.asarray(v[i].ridge_vertices[j])
                pointidx = v[i].ridge_points[j]
                if np.all(simplex >= 0):
                    v_lines[j].set_data(v[i].vertices[simplex, 0], v[i].vertices[simplex, 1])
                else:
                    #center = points.mean(axis=0)
                    #e = simplex[simplex >= 0][0] # finite end voronoi vertex
                    #i = simplex[simplex >= 0][0] # finite end Voronoi vertex
                    #n = np.array([-t[1], t[0]]) # normal
                    #midpoint = points[pointidx].mean(axis=0)
                    # the far point is going to be along this line intersecting with the boundary of the environment
                    #far_point = vor.vertices[e] + np.sign(np.dot(vor.vertices[e] - center, n)) * n * 100
                    #far_point = vor.vertices[e] + np.sign(np.dot(vor.vertices[e] - center, n)) * n * 10
                    #voronoi_lin_segs[i].set_data([vor.vertices[e,0], far_point[0]],
                    #[vor.vertices[e,1], far_point[1]])
                    v_lines[j].set_data([],[])
        
    return animate


