import scipy.integrate as integrate
from scipy.stats import norm
import numpy as np
from math import e

def estimate_shap(weights, bias):
    # Given a voting game of n players return approx shapley values
    # weights: np.array of size n
    # returns an np.array of size n
    #print (weights)
    n = len(weights) # number of players
    q = bias # quota for the game
    mean = np.mean(weights)
    print ("Mean: ", mean)
    var = np.var(weights)
    print ("Var: ", var)

    Xs = range(0, n, max(1, int(n / 10)))
    shapley = []

    def relu(x):
        return x * (x>0)

    def integrand(x, X, wi):
        return e**(-X * (x-mean)**2 / (2*var)) * (relu(x*X + q + wi) - relu(x*X + q))

    for wi in weights:
        r = 0.0
        for X in Xs:
            if X == 0:
                r += (relu(q + wi) - relu(q))
            # elif X == 1:
            #     r += (max(0, q + mean + wi) - max(0, q + mean))
            else:
                #r += (relu(q + wi) - relu(q))
                const = 1. / (np.sqrt(2*np.pi*var/X))
                r += const * integrate.quad(integrand, -np.inf, np.inf, (X,wi))[0]

                #r += (1.0 - norm.cdf(b, mean, np.sqrt(var))) * wi
        shapley.append(r / len(Xs))
    return np.array(shapley)



def main():
    x = np.random.rand(768)
    w = 0.01 * np.random.rand(768)
    b = 0.01 * np.random.rand(1)
    #b = 200
    print("x: ", x)
    print("b: ", b)
    print("x * w: ", x * w)
    print("y: ", max(0, np.sum(x*w)+b))
    shapley = estimate_shap(x*w, b)
    print("Shapley: ", shapley)
    print("Sum: ", np.sum(shapley))

if __name__ == "__main__":
    main()
