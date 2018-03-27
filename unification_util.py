import numpy as np

def mixing_function(alpha):
    def g (f_vec):
        s = np.zeros(f_vec[0].shape)
        for f in f_vec:
            s += np.power(f, alpha)
        return np.power(s, 1.0 / alpha)
    return g 

# appropriate for light based sensors as light intensity drops 
# off as the inverse square from the source 
def quadratic_f(pi, mesh):
    assert(pi.shape==(2,))
    X, Y = mesh
    return 0.5 * np.power( np.power(pi[0] - X, 2) + np.power(pi[1] - Y, 2) , 1.0)

def grad_quadratic_f(pi, mesh):
    X, Y = mesh
    grad_x = pi[0] - X
    grad_y = pi[1] - Y
    return grad_x, grad_y

## Proportial to distance travelled (more appropriate for 
# robots who have to service a point q
def linear_f(pi, q):
    return np.absolute(np.linalg.norm(q - pi))

def grad_linear_r(pi, q):
    return 1

# potential field herding importance function
# delta dirac 
def delta(alpha, x):
    return 1.0 / alpha / np.sqrt(np.pi) * np.exp(- np.power(x / alpha, 2))


def uniform_importance_func():
    def phi(mesh):
        X, Y = mesh
        Z = np.ones(X.shape)
        return 0.2 * Z
    return phi

def gaussian_importance_func(center=(0.5, 0.5)):
    def phi(mesh):
        X, Y = mesh
        Z = np.exp(-np.square(center[0]-X) - np.square(center[0]-Y))
        return Z
    return phi

def multivariate_gaussian_importance_func(centers):
    def phi(mesh):
        X, Y = mesh
        return sum ([np.exp(-np.square(c[0]-X)-np.square(c[1]-Y)) for c in centers])
    return phi

resolution = 100;

def make_H(P, sensing_fn, mix_fn, alpha, imp_fn, 
        x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, resolution=100): 
    g = mix_fn(alpha)
    phi = imp_fn(1.0, P)
    
    def grad_h_pi(pi):
        # integrate this fn over all points 
        x = np.linspace(x_min, x_max, resolution)
        y = np.linspace(y_min, y_max, resolution)
        X,Y = np.meshgrid(x,y)
        
        a = np.power( f(pi, q) / g(sensing_fn(P)), alpha - 1) * (q - pi) * phi(q)
