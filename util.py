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
    return np.random.randn(n, 2) * r + center

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

"""
 	phi: some function we want to integrate 
	center: the center of the circular bounding region
	radius: radius of the circular region we are integrating
	vor_ridges: a list of lines representing a voronoi partition
"""
def integrate_r_limited_voronoi(phi, point_list, point_index, center, radius, vor, resolution):
	x_range = np.linspace(center[0] - radius, center[0] + radius, resolution)
	y_range = np.linspace(center[1] - radius, center[1] + radius, resolution)
	X,Y = np.meshgrid(x_range, y_range)
	Z = phi(X, Y)
	lines = [] 
	for i in range(len(vor.ridge_vertices)):
		simplex = np.asarray(vor.ridge_vertices[i])
		pointidx = vor.ridge_points[i]
		if point_index in pointidx:
			if np.all(simplex >= 0):
				p1 = vor.vertices[simplex[0]]
				p2 = vor.vertices[simplex[1]]
				lines.append([p1[0], p1[1], p2[0], p2[1]]) 
			else: 
				p1 = vor.vertices[simplex[simplex >= 0][0]]
				p2 = 0.5 * (point_list[pointidx[0]] + point_list[pointidx[1]])
				lines.append([p1[0], p1[1], p2[0], p2[1]])
	is_inside = inside_r_limited_voronoi(center, radius, lines, [X, Y])
	# print Z
	Z *= is_inside
	dx = float(radius * 2) / float(resolution)
	dy = float(radius * 2) / float(resolution)
	return np.sum(Z) * dx * dy

def inside_r_limited_voronoi(center, radius, lines, mesh):
	# inside r radius
	X, Y = mesh
	result = np.ones(X.shape)
	in_circle = inside_circle(center, radius, mesh) # calculate distance @ each point in meshgrid
	result *= in_circle # zero out ones outside radius  
	# on same side of each voronoi line 
	for line in lines:
		is_on_same_side = same_side(center, line, mesh)
		result *= is_on_same_side
	return result

def inside_circle(center, radius, mesh):
	X, Y = mesh 
	return np.square(X - center[0]) + np.square(Y - center[1]) < radius**2

def same_side(point, line, mesh):
	X, Y = mesh
	x1, y1, x2, y2 = line
	ax, ay = point 
	return ((y1 - y2)*(ax - x1)+(x2 - x1)*(ay - y1)) * ((y1 - y2)*(X - x1)+(x2 - x1)*(Y - y1)) >= 0

