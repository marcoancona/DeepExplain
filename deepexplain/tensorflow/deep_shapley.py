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
    mean_b = np.mean(baseline)
    print ("Mean: ", mean)
    var = np.var(weights)
    print ("Var: ", var)
    print ("Bias: ", bias)
    print ("Input: ", np.sum(weights) + bias)

    Xs = range(0, n, max(1, int(n / 30)))
    shapley = []

    def relu(x):
        if x > 0:
            return x
        return 0.0
        #y = x * (x>0)
        #return y[0]
        #return np.tanh(x)

    def integrand(x, X, wi, bi, q):
        return 1. / ((2*3.1415926535*var/X)**0.5) * e**(-X * (x-mean)**2 / (2*var)) * (np.maximum(0, x*X + (n-X-1)*mean_b + q + wi) - np.maximum(0, x*X + (n-X-1)*mean_b + q + bi))
               #* (relu(x*X + q + wi) - relu(x*X + q + bi))


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
                int_range = np.linspace(mean-5*sigma, mean+5*sigma, 2**6+1)
                samples = integrand(int_range, X,wi,bi, q)
                i = np.sum(samples) * (int_range[1]-int_range[0])
                #print (samples)
                #i = integrate.romb(samples, int_range[1]-int_range[0])
                #i = integrate.quad(integrand, mean-4*sigma, mean+4*sigma, (X,wi,bi))[0]
                #print (i)
                #print (wi)
                #assert i < (wi +  0.01)
                r += i


                #r += (1.0 - norm.cdf(b, mean, np.sqrt(var))) * wi
        shapley.append(r / len(Xs))
    return np.array(shapley)



def main():
    x =  np.random.rand(10)
    w =  np.random.rand(10)
    b = np.random.rand(1) -2
    #b = 200
    print("x: ", x)
    print("b: ", b)
    print("x * w: ", x * w)
    print("y: ", max(0, np.sum(x*w)+b))
    shapley = estimate_shap(x*w, b)
    print("Shapley: ", shapley)
    print("Sum: ", np.sum(shapley))
    print (shapley /(x*w))

if __name__ == "__main__":
    main()
