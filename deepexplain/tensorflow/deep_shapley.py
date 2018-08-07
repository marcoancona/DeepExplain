import scipy.integrate as integrate
from scipy.stats import norm
import numpy as np
from math import e
from deepexplain.tensorflow.exact_shapley import compute_shapley

def estimate_shap(weights, bias, baseline):
    """

    :param weights: np.array (n, m), where weights[:, j] are the players for target unit j
    :param bias: np.array (m,)
    :param baseline: same as weights
    :return:
    """

    # print ("Warning, overriding baseline")
    # baseline = np.zeros_like(weights)

    print ("Estimating shape. Input shape: ", weights.shape)

    # Sanity checks
    assert len(weights.shape) == 2
    assert len(bias) == weights.shape[-1]

    n, m = weights.shape

    # Work on delta input (in other words, move value for absent player to its baseline value)
    # Will compute mean and variance on deltas, not on players
    deltas = weights - baseline
    # Compute total bias, which consists of original bias (network weight) and the input due to baseline
    bias_total = bias + np.sum(baseline, 0)

    means = np.sum(deltas, 0)[np.newaxis, ...].repeat(n, 0) / (n-1) - deltas / (n-1)
    #assert means.shape[0] == m
    assert means.shape == (n, m)

    vars = np.sum(deltas**2, 0)[np.newaxis, ...].repeat(n, 0)- deltas**2
    _sum = np.sum(deltas, 0)[np.newaxis, ...].repeat(n, 0)- deltas
    vars = (vars - _sum / (n-1)) / (n-2 if n-2 > 0 else np.inf)
    vars = vars + 0.001

    #vars = np.sum((weights - means)**2, 0)[np.newaxis, ...].repeat(n, 0) / (n - 2) - (weights - means)**2 / (n - 2)
    #assert vars.shape == (n, m)

    #print (means)
    #print (means_b)
    #print (vars)

    #assert means.shape[0] == m
    #assert means_b.shape[0] == m
    #assert vars.shape[0] == m

    I =  baseline + deltas + np.repeat(np.expand_dims(bias_total, 0), n, 0)
    Ib = baseline + np.repeat(np.expand_dims(bias_total, 0), n, 0)

    assert I.shape == (n, m)
    assert Ib.shape == (n, m)

    Xs = range(0, n, max(1, int(n / 20)))
    steps = 2**8+1
    R = np.zeros_like(I)


    def integrand(t, k):
        """
        :param t: variable of integration - sample average
        :param k: number of players in the coalition (scalar)
        :return: 2-d numpy array of integrand value. Size (n, m)
        """
        const = 1. / ((2*3.1415926535*vars/k)**0.5)
        const = np.expand_dims(const, -1)
        assert const.shape == (n, m, 1), const.shape
        exp = e**(-k * (t-means[..., np.newaxis])**2 / (2*vars[..., np.newaxis]))
        assert exp.shape == (n, m, steps), exp.shape
        #exp = np.expand_dims(exp, 0)
        #assert exp.shape == (n, m, steps)
        gain = np.maximum(0, np.expand_dims(I, -1) + t*k) - np.maximum(0, np.expand_dims(Ib, -1) + t*k)
        assert gain.shape == (n, m, steps), gain.shape
        # [n, m, 1] * [m, m, steps] * [n, m, steps]
        return const * exp * gain

    for k in Xs:
        print(str(k)+"/"+str(n), end='\r')
        if k == 0:
            R += (np.maximum(0, I) - np.maximum(0, Ib))
            #print ('R: ', R)
        else:
            int_range = np.linspace(-5, 5, steps)
            delta = (int_range[1]-int_range[0])
            s = integrand(int_range + delta/2, k)
            assert s.shape == (n , m, steps)
            R += np.sum(s, -1) * delta

    return R / len(Xs)


def eta_shap(weights, bias, baseline=None):
    if baseline is None:
        baseline = np.zeros_like(weights)

    # return eta_shap_exact(weights, bias, baseline)
    #return eta_shap_dl(weights, bias, baseline)

    shap = estimate_shap(weights, bias, baseline)
    divisor = weights - baseline
    result = np.divide(shap, divisor, out=np.zeros_like(shap), where=divisor!=0)
    return result


def eta_shap_exact(weights, bias=None, baseline=None, f='relu'):
    #print ("Eta shap exact")
    #print (weights)
    n, m = weights.shape
    shap = np.zeros_like(weights)
    if baseline is None:
        baseline = np.zeros_like(weights)
    if bias is None:
        bias = np.zeros_like((m,))
    for i in range(m):
        if f is 'relu':
            f_ = lambda x: np.maximum(np.sum(x) + bias[i], 0)
        else:
            f_ = f
        shap[:, i] = compute_shapley(weights[:, i], f_, baseline=baseline[:, i])
    divisor = weights - baseline
    result= np.divide(shap, divisor, out=np.zeros_like(shap), where=divisor != 0)
    #print (result)
    return result



def eta_shap_dl(weights, bias, baseline=None):
    # Try to replicate DeepLIFT (RevealCancel)

    n, m = weights.shape

    def f(w, b):
        return np.maximum(w + b, 0)

    dx_p = np.maximum((weights - baseline), 0).sum(0)
    dx_m = np.minimum((weights - baseline), 0).sum(0)
    baseline = baseline.sum(0)

    dy_plus = 0.5 * (f(baseline + dx_p, bias) - f(baseline, bias)) + \
              0.5 * (f(baseline + dx_p + dx_m, bias) - f(baseline + dx_m, bias))
    dy_minus = 0.5 * (f(baseline + dx_m, bias) - f(baseline, bias)) + \
              0.5 * (f(baseline + dx_m + dx_p, bias) - f(baseline + dx_p, bias))

    assert dy_plus.shape == (m, ), dy_plus.shape


    eta_plus = np.divide(dy_plus, dx_p, out=np.zeros_like(dy_plus), where=dx_p!=0)
    eta_minus = np.divide(dy_minus, dx_m, out=np.zeros_like(dy_minus), where=dx_m!=0)
    eta = np.where((weights - baseline)> 0, eta_plus, eta_minus)

    assert eta.shape == (n, m), eta.shape
    assert np.all(eta >= 0)
    assert np.all(eta <= 1.0001)
    return eta


def main():
    x =  np.random.rand(3)
    w =  np.random.rand(3, 4)
    b = np.random.rand(4) + 0

    n = 5
    x = np.array([-1.0] + [2/(n-1)]*(n-1))
    w = np.array([1.0] * n)[...,np.newaxis]
    b = np.array([0.0])
    # print(x.shape)
    # print(w.shape)
    # print(b.shape)
    #
    #
    # print("x: ", x)
    # print("b: ", b)
    outer = np.expand_dims(x, 1) * w
    #print("Outer (players): ", outer)
    #print("Outer (shape): ", outer.shape)
    y = np.maximum(0, np.sum(outer, 0)+b)
    #print("y: ", y)
    eta = eta_shap(np.expand_dims(x, 1) * w, b)
    print ('Shapley:', outer * eta)
    print ('Checksum: the following should be similar')
    print (y - b)
    print (np.sum(outer * eta, 0))


if __name__ == "__main__":
    main()
