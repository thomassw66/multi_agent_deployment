#!/usr/bin/env python

import numpy as np
import argparse
import time 

from unification_util import *
from plot_trajectory import * 
from util import *

def coords(s):
    try:
        x, y = map(float, s.split(','))
        return x, y
    except:
        raise argparse.ArgumentTypeError("coordinates must be x,y")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="unified geometric, probablistic, and potential field deployment")
    parser.add_argument("--time_steps", type=int, default=100, help='# of timesteps to sim')
    parser.add_argument("--num_agents", type=int, default=20, help='# agents to sim')
    parser.add_argument("--resolution", type=int, default=50, help='integration resolution')
    parser.add_argument("--alpha", type=float, default=-1.0, help='alpha value to parameterize mixing function')
    parser.add_argument("--step_size", type=float, default=0.05)
    parser.add_argument("--random_seed", type=int, default=42, help="random seed")
    parser.add_argument("--agent_center", type=coords, default=(0,0))    
    parser.add_argument("--agent_variance", type=float, default=1.0)
    parser.add_argument("--importance_centers", type=coords, default=[(0,0)], nargs='+')
    parser.add_argument("--boundaries", type=coords, default=[(-1, 1), (-1, 1)], nargs=2)
    parser.add_argument("--plot_vor", type=bool, default=False)
    parser.add_argument("--run_id", default=time.strftime("%Y_%m_%d-%H_%M_%S"))
    parser.add_argument("--noisiness", type=float, default=0.0)

    args = parser.parse_args()
     
    run_id = args.run_id
    TIME_STEPS = args.time_steps
    num_agents = args.num_agents
    x_min, x_max = args.boundaries[0]
    y_min, y_max = args.boundaries[1]
    res = args.resolution
    importance_centers = np.asarray(args.importance_centers)

    step_size = args.step_size
    noisiness = args.noisiness

    agent_center=args.agent_center
    agent_variance=args.agent_variance
    agents = make_random_points(num_agents, agent_variance, agent_center) 

    random_seed = args.random_seed
    np.random.seed(random_seed)

    dx = float(x_max - x_min) / float(res)
    dy = float(y_max - y_min) / float(res)
    
    x = np.linspace(x_min, x_max, res)
    y = np.linspace(y_min, y_max, res)
    X, Y = np.meshgrid(x, y)
    mesh = X,Y
    
    alpha = args.alpha
    f = quadratic_f
    grad_f = grad_quadratic_f
    phi = multivariate_gaussian_importance_func(importance_centers)        
    eval_H, grad_H = make_grad_H(alpha, f, grad_f, phi)
   
 
    # ******* Generate trajectories    
    x_t = [agents]
    H_t = [eval_H(agents, mesh, dx, dy)]
    for i in range(1, TIME_STEPS):
        x = x_t[-1].copy()
        H = grad_H(x, mesh, dx, dy)

        noise = np.random.randn(num_agents, 2)
        noise_normals = np.array(map(np.linalg.norm, noise)).reshape(num_agents, 1)
        noise = np.divide(noise, noise_normals)
        noise = (noisiness) * noise
        
        H_normals = np.array(map(np.linalg.norm, H)).reshape(num_agents, 1)
        H += np.multiply(H_normals, noise)
        H = np.divide(H, H_normals)

        x -= (step_size) * H
        x_t.append(x)
        H_t.append(eval_H(x, mesh, dx, dy)) # record cost 
    x_t = np.array(x_t) # had to wait to cast so we could use array.append
    H_t = np.array(H_t)

    np.savez("simulations/" + run_id + ".npz", x_t=x_t, H_t=H_t, bound=[x_min, x_max, y_min, y_max], mesh=mesh, Z=-phi(mesh))
    print "done"

    # ************* SHOW TRAJECTORIES ***********************************
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    
    plt.figure(1)
    plt.plot(range(len(H_t)), H_t, 'r--')

    fig, ax = plt.subplots()
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    # visualize importance function
    plt.contour(X,Y, -phi(mesh), 100, cmap='pink')
    # animate trajectories
    animate = plot_trajectory_animation( ax, x_t, plot_vor=args.plot_vor)
    ani = FuncAnimation(fig, animate, frames=TIME_STEPS, interval=2000.0/TIME_STEPS)
    plt.show()
