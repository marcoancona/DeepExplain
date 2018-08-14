import scipy.integrate as integrate
from scipy.stats import norm
import numpy as np
from math import e
import tensorflow as tf
from deepexplain.tensorflow.exact_shapley import compute_shapley


def eta_shap(games, bias=None, baseline=None, method='approx', fun='relu'):
    """
    Get shapley eta for a input array of shape [batch, n, m], where n is the number of players
    and m in the number of games the players take part in. In practice, n is the size of one layer
    and m is the size of the layer that follows.
    In this array, games[b, i, j] is the contribution of player i (from one layer)
    in the activation of target j (in the next layer). In other words, games[b, :, j]
    are the players involved in the activation of target j.

    :param games: ndarray [batch, n, m]
    :param bias: ndarray [m]
    :param baseline: ndarray [n, m] or [batch, n, m] (default zero baseline)
    :param method: one of 'approx' (default), 'exact', 'revcancel'
    :param fun: non-linear function to target for Shapley values (default relu)
    :return eta shap: ndarray [batch, n]
    """
    assert len(games.shape) == 3, games.shape
    b, n, m = games.shape
    if baseline is None:
        baseline = np.zeros((n, m))
    else:
        assert baseline.shape == (n, m) or baseline.shape == (b, n, m), baseline.shape
    if bias is None:
        bias = np.zeros((m,))
    else:
        assert bias.shape == (m,), bias.shape

    print ('Eta shap [%s] games[%s], bias[%s], baseline[%s]' %
           (method, games.shape, bias.shape, baseline.shape))

    # Interpretation: we have n players taking part in b games of m rounds each.
    # For each game b, the Shapley value of each player is the sum of the m (weighted) rounds

    # Reshape to get (b*m) games, each with n players --> (n, b*m)
    games = np.reshape(np.transpose(games, (1, 0, 2)), (n, b*m))
    # Need to repeat baseline and bias!
    bias = np.repeat(bias, b, -1)
    if baseline.shape == (b, n, m):
        baseline = np.reshape(np.transpose(baseline, (1, 0, 2)), (n, b*m))
    else:
        baseline = np.repeat(baseline, b, -1)

    print ('Reshape: games[%s], bias[%s], baseline[%s]' %
           (games.shape, bias.shape, baseline.shape))

    assert baseline.shape == games.shape, baseline.shape

    if method == 'approx':
        eta = eta_shap_approx(games, bias, baseline)
    elif method == 'exact':
        eta = eta_shap_exact(games, bias, baseline, fun=fun)
    elif method == 'revcancel':
        eta = eta_shap_dl(games, bias, baseline)
    else:
        raise RuntimeError('Method eta_shap called with invalid method name [%s]' % (method,))

    assert eta.shape == (n, b*m), eta.shape
    # Reshape back
    result = tf.transpose(tf.reshape(eta, (n, b, m)), (1, 0, 2))
    assert result.shape == (b, n, m), result.shape
    return result


def eta_shap_approx(weights, bias, baseline):
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
    #bias_total = bias + np.sum(baseline, 0)

    means = np.sum(deltas, 0)[np.newaxis, ...].repeat(n, 0) / (n-1) - deltas / (n-1)
    #assert means.shape[0] == m
    assert means.shape == (n, m)

    vars = np.sum(deltas**2, 0)[np.newaxis, ...].repeat(n, 0)- deltas**2
    _sum = np.sum(deltas, 0)[np.newaxis, ...].repeat(n, 0)- deltas
    vars = (vars - _sum / (n-1)) / (n-2 if n-2 > 0 else np.inf)
    vars = np.maximum(0.1, vars) # TODO: check this, as the np.all(vars>0) fails sometimes without max
    assert vars.shape == (n, m)
    assert np.all(vars>0)
    #vars = np.sum((weights - means)**2, 0)[np.newaxis, ...].repeat(n, 0) / (n - 2) - (weights - means)**2 / (n - 2)


    I =  baseline + deltas + np.repeat(np.expand_dims(bias, 0), n, 0)
    Ib = baseline + np.repeat(np.expand_dims(bias, 0), n, 0)

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
        assert k>0
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
        if k == 0:
            R += (np.maximum(0, I) - np.maximum(0, Ib))
            #print ('R: ', R)
        else:
            int_range = np.linspace(-5, 5, steps)
            delta = (int_range[1]-int_range[0])
            s = integrand(int_range + delta/2, k)
            assert s.shape == (n , m, steps)
            R += np.sum(s, -1) * delta

    shap = R / len(Xs)
    divisor = weights - baseline
    return np.divide(shap, divisor, out=np.zeros_like(shap), where=divisor!=0)


def eta_shap_exact(weights, bias=None, baseline=None, fun='relu'):
    #print ("Eta shap exact")
    #print (weights)
    n, m = weights.shape
    shap = np.zeros_like(weights)

    for i in range(m):
        if fun is 'relu':
            f_ = lambda x: np.maximum(np.sum(x, -1) + bias[i], 0)
        else:
            f_ = fun
        shap[:, i] = compute_shapley(weights[:, i], f_, baseline=baseline[:, i])
    divisor = weights - baseline
    result= np.divide(shap, divisor, out=np.zeros_like(shap), where=divisor != 0)
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
    assert np.all(eta <= 1.01)
    return eta


def main():
    b, n, m = 10, 6, 200
    x = np.random.rand(b, n) # (batch, #input)
    w = np.random.rand(n, m) # (#input, #input2)
    b = np.random.rand(m) + 5  # (#input2)

    # n = 5
    # x = np.array([-1.0] + [2/(n-1)]*(n-1))
    # w = np.array([1.0] * n)[...,np.newaxis]
    # b = np.array([0.0])
    # print(x.shape)
    # print(w.shape)
    # print(b.shape)
    #
    #
    print("x: ", x)
    # print("b: ", b)
    session = tf.Session()
    games = np.expand_dims(x, -1) * w
    print("Games (players): ", games)
    #print("Outer (shape): ", outer.shape)
    y = np.maximum(0, np.sum(games, 1)+b)
    print("y: ", y)

    print ("Shap Exact")
    eta_exact = eta_shap(games, bias=b, method='exact').eval(session = session)
    print ("Eta shape:", eta_exact.shape)
    print ('Eta exact:', eta_exact)
    print ('Checksum: the following should be similar')
    print (y - b)
    print (np.sum(x * np.sum((np.expand_dims(w, 0) * eta_exact), -1), -1))

    print ("Shap RevCanccel")
    eta_revcancel = eta_shap(games, bias=b, method='revcancel').eval(session=session)
    print ("Eta shape:", eta_revcancel.shape)
    print ('Eta revcancel:', eta_revcancel)
    print ('Checksum: the following should be similar')
    print (y - np.maximum(0, b))
    print (np.sum(x * np.sum((np.expand_dims(w, 0) * eta_revcancel), -1), -1))

    print ("Shap Approx")
    eta_approx = eta_shap(games, bias=b, method='approx').eval(session = session)
    print ("Eta shape:", eta_approx.shape)
    print ('Eta apprx:', eta_approx)
    print ('Checksum: the following should be similar')
    print (y - np.maximum(0, b))
    print (np.sum(x * np.sum((np.expand_dims(w, 0) * eta_approx), -1), -1))

    print ("Approx error")
    print (np.sqrt(np.mean((eta_exact-eta_approx)**2)))
    print ("Revcancel error")
    print (np.sqrt(np.mean((eta_exact - eta_revcancel) ** 2)))




if __name__ == "__main__":
    main()
