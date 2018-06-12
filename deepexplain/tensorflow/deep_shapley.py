import scipy.integrate as integrate
from scipy.stats import norm
import numpy as np
from math import e

def estimate_shap(weights, bias, baseline=None):
    # Given a voting game of n players return approx shapley values
    # weights: np.array of size n
    # returns an np.array of size n
    #print (weights)
    if baseline is None:
        baseline = np.zeros_like(weights)
    n = len(weights) # number of players
    q = bias # quota for the game
    mean = np.mean(weights)
    print ("Mean: ", mean)
    var = np.var(weights)
    print ("Var: ", var)

    Xs = range(0, n, max(1, int(n / 30)))
    shapley = []

    def relu(x):
        return x * (x>0)

    def integrand(x, X, wi, bi):
        return 1. / ((2*3.1415926535*var/X)**0.5) * e**(-X * (x-mean)**2 / (2*var)) * (relu(x*X + q + wi) - relu(x*X + q))

    for wi, bi in zip(weights, baseline):
        r = 0.0
        for X in Xs:
            if X == 0:
                r += (relu(q + wi) - relu(q + bi))
            # elif X == 1:
            #     r += (max(0, q + mean + wi) - max(0, q + mean))
            else:
                #r += (relu(q + wi) - relu(q))
                sigma = (var/X)**0.5
                r += integrate.fixed_quad(integrand, mean-4*sigma, mean+4*sigma, (X,wi,bi))[0]

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
