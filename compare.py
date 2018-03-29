import sys
import numpy as np

if __name__ == "__main__":
    if (len(sys.argv) < 3):
        print "usage: compare.py run_id run_id"

    run_id1 = sys.argv[1]
    run_id2 = sys.argv[2]
    
    data_1 = np.load("simulations/" + run_id1 + ".npz")
    data_2 = np.load("simulations/" + run_id2 + ".npz")

    H_t1 = data_1["H_t"]
    H_t2 = data_2["H_t"]

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    
    plt.figure(1)
    plt.plot(range(len(H_t2)), H_t1, 'r--', range(len(H_t2)), H_t2, 'b--')
    plt.show()


