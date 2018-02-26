import numpy as np


def ellipsoid(X, Y, a, b, center=(0,0)):
    return np.sqrt(np.square(X - center[0]) / a**2 + np.square(Y - center[1]) / b**2)

def circle(X, Y, center=(0,0)):
    return np.sqrt(np.square(X - center[0]) + np.square(Y - center[1]))
