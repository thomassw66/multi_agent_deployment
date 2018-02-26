import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


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

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


# creates n uniformly distributed random points
def make_random_points(n, r, center=(0,0)):
    return np.random.random([n, 2]) * r + center

# given a convex polygon defining a voronoi partition compute the
# integral
# PHI HAS TO RETURN A MATRIX s.t. size of X and Y  == size of Z
def approx_integrate_poly(phi, points, resolution):
    x_low = min(points[:,0])
    x_high = max(points[:,0])
    y_low = min(points[:,1])
    y_high = max(points[:,1])
    x = np.linspace(x_low, x_high, resolution)
    y = np.linspace(y_low, y_high, resolution)
    X,Y = np.meshgrid(x,y)
    Z = phi(X,Y)
    # for points not inside polygon set Z to zero
    # TODO: make this less excruciatingly slow (not feasible to compute for resolution > 100)
    polygon = Polygon(points)
    is_in_poly = np.ones((resolution, resolution))
    for i in range(resolution):
        for j in range(resolution):
            p = Point(X[i][j],Y[i][j])
            is_in_poly[i][j] = polygon.contains(p) + polygon.touches(p)
    # This polygon library is going to kill me ^
    dx = float(x_high - x_low) / float(resolution)
    dy = float(y_high - y_low) / float(resolution)
    area = dx * dy
    return area * np.sum(Z * is_in_poly)

