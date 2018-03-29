import numpy as np
from unification_util import *
from plot_trajectory import * 
from util import *

if __name__ == "__main__":
    TIME_STEPS = 200

    num_agents = 20
    x_min, x_max = 0.0, 5.0
    y_min, y_max = 0.0, 5.0
    res = 50

    dx = float(x_max - x_min) / float(res)
    dy = float(y_max - y_min) / float(res)
    
    x = np.linspace(x_min, x_max, res)
    y = np.linspace(y_min, y_max, res)
    X, Y = np.meshgrid(x, y)
    mesh = X,Y
    
    alpha = -1.0
    f = quadratic_f
    grad_f = grad_quadratic_f
    phi = multivariate_gaussian_importance_func([ (3.0, 0.5)])        
    eval_H, grad_H = make_grad_H(alpha, f, grad_f, phi)
    np.random.seed(42)
    agents = make_random_points(num_agents, 1.5, center=(2.5,2.5)) 
    # ******* Generate trajectories    
    x_t = [agents]
    H_t = [eval_H(agents, mesh, dx, dy)]
    for i in range(1, TIME_STEPS):
        x = x_t[-1].copy()
        H = grad_H(x, mesh, dx, dy)
        print eval_H(x, mesh, dx, dy)
        # add some normally distributed noise to our gradient vector? 
        noise = np.random.randn(num_agents, 2)
         
        for j in range(len(H)):
            norm = np.linalg.norm(H[j])
            if norm > 0.0: 
                H[j] = H[j] / norm
                H[j] = H[j] + noise[j] * norm 
        x -= (0.05) * H
        print (i) 
        x_t.append(x)
        H_t.append(eval_H(x, mesh, dx, dy))
    x_t = np.array(x_t) # had to wait to cast so we could use array.append
    H_t = np.array(H_t)

    # ************* SHOW TRAJECTORIES ***********************************
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    plt.figure(1)
    plt.plot(range(0, TIME_STEPS), H_t, 'r--')

    fig, ax = plt.subplots()
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    # visualize importance function
    plt.contour(X,Y, -phi(mesh), 100, cmap='pink')
    # animate trajectories
    animate = plot_trajectory_animation(ax, x_t)
    ani = FuncAnimation(fig, animate, frames=TIME_STEPS, interval=2000.0/TIME_STEPS)
    plt.show()
