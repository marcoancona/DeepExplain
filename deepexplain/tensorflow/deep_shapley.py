import scipy.integrate as integrate
from scipy.stats import norm
import numpy as np
from math import e

def estimate_shap(weights, bias, baseline):
    """

    :param weights: np.array (n, m), where weights[:, j] are the players for target unit j
    :param bias: np.array (m,)
    :param baseline: same as weights
    :return:
    """

    print ("Estimating shape. Input shape: ", weights.shape)

    # Sanity checks
    assert len(weights.shape) == 2
    assert len(bias) == weights.shape[-1]

    n, m = weights.shape

    means = np.mean(weights, 0)
    means_b = np.mean(baseline, 0)
    vars = np.var(weights, 0)

    assert means.shape[0] == m
    assert means_b.shape[0] == m
    assert vars.shape[0] == m

    I = weights + np.repeat(np.expand_dims(bias, 0), n, 0)
    Ib = baseline + np.repeat(np.expand_dims(bias, 0), n, 0)

    assert I.shape == (n, m)
    assert Ib.shape == (n, m)

    Xs = range(0, n, max(1, int(n / 30)))
    steps = 2**6+1
    R = np.zeros_like(I)


    def integrand(t, k):
        """
        :param t: variable of integration - sample average
        :param k: number of players in the coalition
        :return: 2-d numpy array of integrand value. Size (n, m)
        """
        const = 1. / ((2*3.1415926535*vars/k)**0.5)
        const = np.expand_dims(const, 1)
        assert const.shape == (m, 1)
        exp = e**(-k * (t-means[..., np.newaxis])**2 / (2*vars[:, np.newaxis]))
        assert exp.shape == (m, steps)
        exp = np.expand_dims(exp, 0)
        assert exp.shape == (1, m, steps)
        gain = np.maximum(0, np.expand_dims(I, -1) + t*k) - np.maximum(0, np.expand_dims(Ib, -1) + t*k)
        assert gain.shape == (n, m, steps)
        # [m, 1] * [m, steps] * [n, m, steps]
        return const * exp * gain

    for k in Xs:
        print(str(k)+"/"+str(n), end='\r')
        if k == 0:
            R += (np.maximum(0, I) - np.maximum(0, Ib))
        else:
            sigma = (np.mean(vars)/k)**0.5  # just a guess
            mean = np.mean(means)
            assert mean.shape == ()
            assert sigma.shape == ()

            int_range = np.linspace(mean-5*sigma, mean+5*sigma, steps)
            s = integrand(int_range, k)
            assert s.shape == (n , m, steps)
            R += np.sum(s, -1) * (int_range[1]-int_range[0])

    return R / len(Xs)


def eta_shap(weights, bias, baseline=None):
    if baseline is None:
        baseline = np.zeros_like(weights)
    shap = estimate_shap(weights, bias, baseline)
    divisor = weights - baseline
    return np.divide(shap, divisor, out=np.zeros_like(shap), where=divisor!=0)


def main():
    x =  np.random.rand(3)
    w =  np.random.rand(3, 4)
    b = np.random.rand(4) + 0
    #b = 200
    print("x: ", x)
    print("b: ", b)
    outer = np.expand_dims(x, 1) * w
    print("Outer (players): ", outer)
    print("Outer (shape): ", outer.shape)
    y = np.maximum(0, np.sum(outer, 0)+b)
    print("y: ", y)
    eta = eta_shap(np.expand_dims(x, 1) * w, b)
    assert eta.shape == (3, 4)
    print ('Checksum: the following should be similar')
    print (y - b)
    print (np.sum(outer * eta, 0))


if __name__ == "__main__":
    main()
