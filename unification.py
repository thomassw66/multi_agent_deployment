import numpy as np
from unification_util import *
from plot_trajectory import * 


def make_grad_H(alpha, f, grad_f, phi):
    g = mixing_function(alpha)
    def grad_H(agents, mesh):
        ga = g([f(a, mesh) for a in agents])
        H = np.zeros((len(agents), 2))
        for i in range(len(agents)):
            f_term = np.power(f(agents[i], mesh) / ga, alpha - 1.0)
            grad_fx, grad_fy = grad_f(agents[i], mesh)
            p_term = phi(mesh)
            H[i,0] = dx * dy * np.sum(f_term * grad_fx * p_term)
            H[i,1] = dx * dy * np.sum(f_term * grad_fy * p_term)
            norm = np.linalg.norm(H[i])
            if norm > 0: H[i] = H[i] / norm
        return H
    return grad_H

if __name__ == "__main__":
    TIME_STEPS = 100

    num_agents = 10
    x_min, x_max = 0.0, 5.0
    y_min, y_max = 0.0, 5.0
    res = 100

    dx = float(x_max - x_min) / float(res)
    dy = float(y_max - y_min) / float(res)
    
    x = np.linspace(x_min, x_max, res)
    y = np.linspace(y_min, y_max, res)
    X, Y = np.meshgrid(x, y)
    mesh = X,Y
    
    alpha = -10
    f = quadratic_f
    grad_f = grad_quadratic_f
    phi = multivariate_gaussian_importance_func([(0.5, 3.5), (3.0, 0.5), (4.0, 4.0)]) 
    g = mixing_function(alpha)

        
    grad_H = make_grad_H(alpha, f, grad_f, phi)

    agents = np.random.random((num_agents, 2)) 
    # ******* Generate trajectories    
    x_t = [agents]
    for i in range(1, TIME_STEPS):
        x = x_t[-1].copy()
        x -= (0.05) * grad_H(x, mesh)
        #for j in range(len(x)): # WARNING use x[j] not x[i]!!!
            # update agents position with gradient descent control
        #    ga = g([f(a, mesh) for a in x]) 
        #    f_term = np.power(f(x[j], mesh) / ga, alpha - 1.0) 
        #    grad_fx, grad_fy = grad_f(x[j], mesh)
        #    p_term = phi(mesh)

        #    gHx = dx * dy * np.sum(f_term * grad_fx * p_term)
        #    gHy = dx * dy * np.sum(f_term * grad_fy * p_term)

        #    grad = np.array([gHx, gHy])
        #    norm = np.linalg.norm(grad)
        #    if norm > 0: grad /= norm
        #    k = 0.15
            # print grad
        #    x[j] -= k * grad
        print (i) 
        x_t.append(x)
    x_t = np.array(x_t) # had to wait to cast so we could use array.append

    # ************* SHOW TRAJECTORIES ***********************************
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    fig, ax = plt.subplots()
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    # visualize importance function
    plt.contour(X,Y, -phi(mesh), 100, cmap='pink')
    # animate trajectories
    animate = plot_trajectory_animation( ax, x_t )
    ani = FuncAnimation(fig, animate, frames=TIME_STEPS, interval=2000.0/TIME_STEPS)
    plt.show()
