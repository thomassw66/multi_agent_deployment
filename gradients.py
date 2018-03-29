import numpy as np
import util

############## COST FUNCTION GRADIENTS #############
# Ciruclar cost function gradient
def circle(point, center=(0,0)):
    return np.array([2*(point[0] - center[0]), 2*(point[1] - center[1])], dtype='float64')

def ellipsoid(point, a, b, center=(0,0)):
    return np.array([2 * (point[0]-center[0]) / a**2 , 2 * (point[1] - center[1]) / b ** 2], dtype='float64')

def gaussian_density(point, center=(0,0)):
    # f = exp(-x^2 - y^2)
    p = point - center
    f = np.exp( -p[0]**2 - p[1]**2)
    return -1 * np.array([ -2*p[0] * f, -2*p[1] * f])

def potential_field(point, neighbors):
    sum = np.array([0, 0], dtype='float64')
    for i in range(len(neighbors)):
        d = neighbors[i] - point
        distance = np.linalg.norm(d)
        sum += (distance ** -2 + 1/3 * distance ** -3) * util.normalize(d)
    return sum


