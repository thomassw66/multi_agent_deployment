from plot_trajectory import *
import numpy as np
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "usage: python visual.py run_id"
        exit(0)

    data = np.load("simulations/" + sys.argv[1] + ".npz")
    x_t = data["x_t"]
    H_t = data["H_t"]
    x_min, x_max, y_min, y_max = data["bound"]
    X, Y = data["mesh"] 
    Z = data["Z"]   
     
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    
    plt.figure(1)
    plt.plot(range(len(H_t)), H_t, 'r--')

    fig, ax = plt.subplots()
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    # visualize importance function
    plt.contour(X,Y, Z, 100, cmap='pink')
    # animate trajectories
    animate = plot_trajectory_animation( ax, x_t)
    ani = FuncAnimation(fig, animate, frames=len(x_t), interval=2000.0/len(x_t))
    plt.show() 
