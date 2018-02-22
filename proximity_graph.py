import numpy as np

# takes a one dimentional array of points [ [x0, y0], [x1, y1], ...] and computes a square distance matrix to each point
def make_proximity_graph(points):
	x = points[:, 0]
	y = points[:, 1]
	dx = x[..., np.newaxis] - x[np.newaxis, ...]
	dy = y[..., np.newaxis] - y[np.newaxis, ...]
	return np.sqrt ( np.square(dx) + np.square(dy) );

# given a distance matrix return 1 if the distance is <= specified radius
def r_disk_graph(prox_graph, r):
    return np.less(prox_graph, r).astype(dtype=int) # this feels more sciency

# return a list of points neighboring point i
def get_neighborhood(i, points, adjacentcy_matrix):
    a = []
    for j in range(len(adjacentcy_matrix[i])):
        if i != j and adjacentcy_matrix[i][j]:
            a.append(points[j])
    return np.array(a)


# points = np.array([[1,1], [2,2], [3,3]])
# pg = make_proximity_graph(points)
# rd = r_disk_graph(pg, 1.5)

# print rd

# for i in range(len(points)):
#   print i
#    print get_neighborhood(i, points, rd)

#
#x = np.linspace(-5, 5, 20);
#y = np.linspace(-5, 5, 20);
#[X, Y] = np.meshgrid(x , y);


