from unification_util import *

if __name__ == "__main__":
    
    x_min, x_max = 0.0, 1.0
    y_min, y_max = 0.0, 1.0
    res = 100
    
    x = np.linspace(x_min, x_max, res)
    y = np.linspace(y_min, y_max, res)
    X, Y = np.meshgrid(x, y)
    mesh = X,Y

    np.random.seed(42)
    agents = np.random.random((20, 2))

    import matplotlib.pyplot as plt
    plt.plot(agents[:,0], agents[:,1], 'o')

    alphas = [1.0] # [-0.5, -1.0, -5.0, -10.0]
    f = quadratic_f    
    print agents    
    
    for i in range(len(alphas)):
        alpha =  alphas[0]
        g = mixing_function(alpha)
         
        Z = np.zeros(X.shape)
        ga = g([f(a, mesh) for a in agents])
        
        for a in agents:
            fpiq = f(a, mesh)
            print fpiq
            Z += np.power( fpiq / ga , alpha - 1.0)
        plt.contour(X,Y,Z, 50)  
    plt.show()
