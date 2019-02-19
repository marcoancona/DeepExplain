import scipy.integrate as integrate
from scipy.stats import norm
import numpy as np
import numpy.ma as ma
from math import e
import tensorflow as tf
import scipy
from scipy.special import erf
from deepexplain.tensorflow.exact_shapley import compute_shapley, compute_shapley_legacy
from keras.models import Sequential
from keras.layers import Dense

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

    #print (baseline)



    assert len(games.shape) == 3, games.shape
    b, n, m = games.shape

    # print ("WARNING: SETTING BASELINE TO ZERO 2")
    # baseline = np.zeros((n, m))

    if baseline is None:
        baseline = np.zeros((n, m))
    elif baseline.shape == (n, m):
        baseline = np.repeat(np.expand_dims(baseline, 0), b, 0)
    else:
        assert baseline.shape == (b, n, m), baseline.shape
    if bias is None:
        bias = np.zeros((m,))
    else:
        assert bias.shape == (m,), bias.shape



    print ('Eta shap [%s] games[%s], bias[%s], baseline[%s]' %
           (method, games.shape, bias.shape, baseline.shape))

    # Interpretation: we have n players taking part in b games of m rounds each.
    # For each game b, the Shapley value of each player is the sum of the m (weighted) rounds

    # Reshape to get (b*m) games, each with n players --> (n, b*m)
    # games = np.reshape(np.transpose(games, (1, 0, 2)), (n, b*m))
    # # Need to repeat baseline and bias!
    # bias = np.repeat(bias, b, -1)
    # if baseline.shape == (b, n, m):
    #     baseline = np.reshape(np.transpose(baseline, (1, 0, 2)), (n, b*m))
    # else:
    #     baseline = np.repeat(baseline, b, -1)

    print ('Reshape: games[%s], bias[%s], baseline[%s]' %
           (games.shape, bias.shape, baseline.shape))

    #assert baseline.shape == games.shape, baseline.shape

    # if method is 'approx':
    #     games = tf.convert_to_tensor(games, dtype='float32')
    #     bias = tf.convert_to_tensor(bias, dtype='float32')
    #     baseline = tf.convert_to_tensor(baseline, dtype='float32')

    result = np.zeros_like(games)
    for idx in range(b):
        if method == 'approx':
            eta = eta_shap_approx_3(games[idx], bias, baseline[idx])
        elif method == 'exact':
            eta = eta_shap_exact(games[idx], bias, baseline[idx], fun=fun)
        elif method == 'revcancel':
            eta = eta_shap_dl(games[idx], bias, baseline[idx])
        elif method == 'rescale':
            eta = eta_shap_rescale(games[idx], bias, baseline[idx])
        else:
            raise RuntimeError('Method eta_shap called with invalid method name [%s]' % (method,))
        result[idx] = eta

    # assert eta.shape == (n, b*m), eta.shape
    # Reshape back
    # result = tf.transpose(tf.reshape(eta, (n, b, m)), (1, 0, 2))

    #assert result.shape == (b, n, m), result.shape
    return tf.convert_to_tensor(result)


def eta_shap_approx(weights, bias, baseline):
    """

    :param weights: np.array (n, m), where weights[:, j] are the players for target unit j
    :param bias: np.array (m,)
    :param baseline: same as weights
    :return:
    """

    # print ("Warning, overriding baseline")
    # baseline = np.zeros_like(weights)

    #print ("Estimating shape. Input shape: ", weights.shape)

    # Sanity checks
    #assert len(weights.shape) == 2
    #assert len(bias) == weights.shape[-1]

    #n, m = weights.get_shape().as_list()
    n, m = weights.shape


    # if type(weights) is np.ndarray:
    #     #print ("Convert to tensor")
    #     weights = tf.convert_to_tensor(weights)
    # if type(bias) is np.ndarray:
    #     bias = tf.convert_to_tensor(bias)
    # if type(baseline) is np.ndarray:
    #     baseline = tf.convert_to_tensor(baseline)



    # print (weights.shape)
    # print (bias.shape)
    # print (baseline.shape)

    # Work on delta input (in other words, move value for absent player to its baseline value)
    # Will compute mean and variance on deltas, not on players
    deltas = weights - baseline
    # Compute total bias, which consists of original bias (network weight) and the input due to baseline
    bias_total = bias + np.sum(baseline, 0)

    means = (np.tile(np.sum(deltas, 0, keepdims=True), [n, 1]) - deltas) / (n-1)
    #assert means.shape[0] == m
    assert means.shape == (n, m)

    vars = (np.tile(np.sum(deltas**2, 0, keepdims=True), [n, 1]) - deltas**2) / (n-1)
    vars -= means**2

    #_sum = np.sum(deltas, 0)[np.newaxis, ...].repeat(n, 0)- deltas
    #vars = (vars - _sum / (n-1)) / (n-2 if n-2 > 0 else np.inf)
    vars = np.maximum(0.0, vars) # TODO: check this, as the np.all(vars>0) fails sometimes without max
    assert vars.shape == (n, m)
    #print (n, m)
    #plt.imshow(vars)
    #plt.show()
    #assert np.all(var>=0), np.min(vars)
    #vars = np.sum((weights - means)**2, 0)[np.newaxis, ...].repeat(n, 0) / (n - 2) - (weights - means)**2 / (n - 2)

    # I[i,j] is the input to neuron j, given by bias + baseline + delta of unit i
    # The total input to neuron j can be computed by adding k*t, where t is the expected delta
    # and t is the number of additional players (on top of i)
    # The baseline is included in bias_total
    I =  deltas + np.tile(np.expand_dims(bias_total, 0), [n, 1])
    Ib = np.tile(np.expand_dims(bias_total, 0), [n, 1])

    assert I.shape == (n, m)
    assert Ib.shape == (n, m)

    Xs = range(0, n, max(1, n // 20))
    steps = 2**6
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
        exp = e**(-k * (t-np.expand_dims(means, -1))**2 / (2*np.expand_dims(vars, -1)))
        assert exp.shape == (n, m, steps), exp.shape
        #exp = np.expand_dims(exp, 0)
        #assert exp.shape == (n, m, steps)
        gain = np.maximum(0.0, np.expand_dims(I, -1) + t*k) - np.maximum(0.0, np.expand_dims(Ib, -1) + t*k)
        assert gain.shape == (n, m, steps), gain.shape
        # [n, m, 1] * [m, m, steps] * [n, m, steps]
        return const * exp * gain

    #_m = np.mean(means)
    #_v = np.mean(np.)**0.5
    for k in Xs:
        if k == 0:
            R += (np.maximum(0.0, I) - np.maximum(0.0, Ib))
            #print ('R: ', R)
        else:
            int_range = np.linspace(-2, 2, steps)
            #int_range = np.linspace(-2, 2, steps)
            delta = int_range[1]-int_range[0]
            s = integrand(int_range, k)
            assert s.shape == (n , m, steps)
            integral = np.trapz(s, int_range) #
            #integral = np.sum(s, -1) * delta
            gain = np.maximum(0.0, I + means * k) - np.maximum(0.0, Ib + means * k)
            #print (gain.shape)
            R += np.where(vars>0.01, integral, gain)

    shap = R / len(Xs)
    eta =  np.where(np.abs(deltas) > 1e-5, shap / deltas, np.zeros_like(shap))
    assert np.all(eta >= -1e-10), np.min(eta)
    assert np.all(eta <= 1.2), np.max(eta)
    #print (eta)
    return eta



def eta_shap_approx_2(weights, bias, baseline):
    """

    :param weights: np.array (n, m), where weights[:, j] are the players for target unit j
    :param bias: np.array (m,)
    :param baseline: same as weights
    :return:
    """

    # print ("Warning, overriding baseline")
    # baseline = np.zeros_like(weights)

    #print ("Estimating shape. Input shape: ", weights.shape)

    # Sanity checks
    #assert len(weights.shape) == 2
    #assert len(bias) == weights.shape[-1]

    #n, m = weights.get_shape().as_list()
    n, m = weights.shape

    # if type(weights) is np.ndarray:
    #     #print ("Convert to tensor")
    #     weights = tf.convert_to_tensor(weights)
    # if type(bias) is np.ndarray:
    #     bias = tf.convert_to_tensor(bias)
    # if type(baseline) is np.ndarray:
    #     baseline = tf.convert_to_tensor(baseline)



    # print (weights.shape)
    # print (bias.shape)
    # print (baseline.shape)

    # Work on delta input (in other words, move value for absent player to its baseline value)
    # Will compute mean and variance on deltas, not on players
    deltas = weights - baseline
    # Compute total bias, which consists of original bias (network weight) and the input due to baseline
    bias_total = bias + np.sum(baseline, 0)

    means = (np.tile(np.sum(deltas, 0, keepdims=True), [n, 1]) - deltas) / (n-1)
    #assert means.shape[0] == m
    assert means.shape == (n, m)

    vars = (np.tile(np.sum(deltas**2, 0, keepdims=True), [n, 1]) - deltas**2) / (n-1)
    vars -= means**2

    #_sum = np.sum(deltas, 0)[np.newaxis, ...].repeat(n, 0)- deltas
    #vars = (vars - _sum / (n-1)) / (n-2 if n-2 > 0 else np.inf)
    #vars = np.maximum(0.0, vars) # TODO: check this, as the np.all(vars>0) fails sometimes without max
    assert vars.shape == (n, m)
    #print (n, m)
    #plt.imshow(vars)
    #plt.show()
    assert np.all(vars>=0), np.min(vars)
    #vars = np.sum((weights - means)**2, 0)[np.newaxis, ...].repeat(n, 0) / (n - 2) - (weights - means)**2 / (n - 2)

    # I[i,j] is the input to neuron j, given by bias + baseline + delta of unit i
    # The total input to neuron j can be computed by adding k*t, where t is the expected delta
    # and t is the number of additional players (on top of i)
    # The baseline is included in bias_total
    I =  deltas + np.tile(np.expand_dims(bias_total, 0), [n, 1])
    Ib = np.tile(np.expand_dims(bias_total, 0), [n, 1])

    assert I.shape == (n, m)
    assert Ib.shape == (n, m)

    Xs = range(0, n, max(1, n // 20))
    Xs = range(0, n)
    R = np.zeros_like(I)


    #_m = np.mean(means)
    #_v = np.mean(np.)**0.5
    pi = 3.1415926535

    for k in Xs:
        if k == 0:
            R += (np.maximum(0.0, I) - np.maximum(0.0, Ib))
        else:
            i = I + k * means
            ib = Ib + k * means
            t = (i / (2  * k * vars)**0.5)
            tb = (ib / (2  * k * vars)**0.5)
            exp1 = e ** -(t ** 2)
            exp2 = e ** -(tb **2)

            t = 0.5*(deltas + (exp1 - exp2) * ((2*k*vars/pi)**0.5) - ib*erf(tb) + i*erf(t))
            assert np.all(np.abs(t) <= 0.01+np.abs(deltas)), (np.min(np.abs(deltas) - np.abs(t)), t.flatten()[np.argmin(np.abs(deltas) - np.abs(t))], deltas.flatten()[np.argmin(np.abs(deltas) - np.abs(t))])
            R += t

    shap = R / len(Xs)
    assert not np.any(np.isnan(shap))
    eta =  np.where(np.abs(deltas) > 1e-6, shap / deltas, np.zeros_like(shap))
    #eta = np.nan_to_num(np.divide(shap, deltas, out=np.zeros_like(shap)))
    assert np.all(eta >= -1e-1), np.min(eta)
    assert np.all(eta <= 1.2), np.max(eta)
    #print (eta)
    return eta



def eta_shap_approx_3(weights, bias, baseline):
    """

    :param weights: np.array (n, m), where weights[:, j] are the players for target unit j
    :param bias: np.array (m,)
    :param baseline: same as weights
    :return:
    """

    # print ("Warning, overriding baseline")
    # baseline = np.zeros_like(weights)

    #print ("Estimating shape. Input shape: ", weights.shape)

    # Sanity checks
    #assert len(weights.shape) == 2
    #assert len(bias) == weights.shape[-1]

    #n, m = weights.get_shape().as_list()
    n, m = weights.shape

    # if type(weights) is np.ndarray:
    #     #print ("Convert to tensor")
    #     weights = tf.convert_to_tensor(weights)
    # if type(bias) is np.ndarray:
    #     bias = tf.convert_to_tensor(bias)
    # if type(baseline) is np.ndarray:
    #     baseline = tf.convert_to_tensor(baseline)



    # print (weights.shape)
    # print (bias.shape)
    # print (baseline.shape)

    # Work on delta input (in other words, move value for absent player to its baseline value)
    # Will compute mean and variance on deltas, not on players
    deltas = weights - baseline
    # Compute total bias, which consists of original bias (network weight) and the input due to baseline
    bias_total = bias + np.sum(baseline, 0)

    means = (np.tile(np.sum(deltas, 0, keepdims=True), [n, 1]) - deltas) / (n-1)
    #assert means.shape[0] == m
    assert means.shape == (n, m)

    vars = (np.tile(np.sum(deltas**2, 0, keepdims=True), [n, 1]) - deltas**2) / (n-1)
    vars -= means**2

    #_sum = np.sum(deltas, 0)[np.newaxis, ...].repeat(n, 0)- deltas
    #vars = (vars - _sum / (n-1)) / (n-2 if n-2 > 0 else np.inf)
    #vars = np.maximum(0.0, vars) # TODO: check this, as the np.all(vars>0) fails sometimes without max
    assert vars.shape == (n, m)
    #print (n, m)
    #plt.imshow(vars)
    #plt.show()
    assert np.all(vars>=0), np.min(vars)
    #vars = np.sum((weights - means)**2, 0)[np.newaxis, ...].repeat(n, 0) / (n - 2) - (weights - means)**2 / (n - 2)

    # I[i,j] is the input to neuron j, given by bias + baseline + delta of unit i
    # The total input to neuron j can be computed by adding k*t, where t is the expected delta
    # and t is the number of additional players (on top of i)
    # The baseline is included in bias_total
    I =  deltas + np.tile(np.expand_dims(bias_total, 0), [n, 1])
    Ib = np.tile(np.expand_dims(bias_total, 0), [n, 1])

    assert I.shape == (n, m)
    assert Ib.shape == (n, m)

    Xs = range(0, n, max(1, n // 20))
    #Xs = range(0, n)
    Xs = np.array(Xs)

    lx = len(Xs)
    #Xs = range(0, n)
    R = np.zeros_like(I)


    #_m = np.mean(means)
    #_v = np.mean(np.)**0.5
    pi = 3.1415926535

    # means, vars [n, m]
    # Xs [n]
    # I, Ib [n, m]

    i = I[:, :, np.newaxis] + means[:, :, np.newaxis] * Xs[np.newaxis, np.newaxis, :]
    ib = Ib[:, :, np.newaxis] + means[:, :, np.newaxis] * Xs[np.newaxis, np.newaxis, :]
    assert i.shape == (n, m, lx), i.shape

    t = (i / (2 * vars[:, :, np.newaxis] * Xs[np.newaxis, np.newaxis, :]) ** 0.5)
    tb = (ib / (2 * vars[:, :, np.newaxis] * Xs[np.newaxis, np.newaxis, :]) ** 0.5)
    t = np.nan_to_num(t)
    tb = np.nan_to_num(tb)
    assert t.shape == (n, m, lx), t.shape


    exp1 = e ** -(t ** 2)
    exp2 = e ** -(tb ** 2)
    assert exp1.shape == (n, m, lx), exp1.shape

    divisor =  ((2 / pi * vars[:, :, np.newaxis] * Xs[np.newaxis, np.newaxis, :])**0.5)
    assert divisor.shape == (n, m, lx), divisor.shape

    R = 0.5*(deltas[:, :, np.newaxis] +  (exp1 - exp2) * divisor - ib*erf(tb) + i*erf(t))
    assert R.shape == (n, m, lx), R.shape

    shap = np.mean(R, -1)
    assert shap.shape == (n, m)

    assert np.all(np.abs(shap) <= 0.01 + np.abs(deltas)), (
    np.min(np.abs(deltas) - np.abs(shap)), shap.flatten()[np.argmin(np.abs(deltas) - np.abs(shap))],
    deltas.flatten()[np.argmin(np.abs(deltas) - np.abs(shap))])

    # for k in Xs:
    #     if k == 0:
    #         R += (np.maximum(0.0, I) - np.maximum(0.0, Ib))
    #     else:
    #         i = I + k * means
    #         ib = Ib + k * means
    #         t = (i / (2  * k * vars)**0.5)
    #         tb = (ib / (2  * k * vars)**0.5)
    #         exp1 = e ** -(t ** 2)
    #         exp2 = e ** -(tb **2)
    #
    #         t = 0.5*(deltas + (exp1 - exp2) * ((2*k*vars/pi)**0.5) - ib*erf(tb) + i*erf(t))
    #         assert np.all(np.abs(t) <= 0.01+np.abs(deltas)), (np.min(np.abs(deltas) - np.abs(t)), t.flatten()[np.argmin(np.abs(deltas) - np.abs(t))], deltas.flatten()[np.argmin(np.abs(deltas) - np.abs(t))])
    #         R += t

    assert not np.any(np.isnan(shap))
    eta =  np.where(np.abs(deltas) > 1e-6, shap / deltas, np.zeros_like(shap))
    #eta = np.nan_to_num(np.divide(shap, deltas, out=np.zeros_like(shap)))
    assert np.all(eta >= -1e-1), np.min(eta)
    assert np.all(eta <= 1.2), np.max(eta)
    #print (eta)
    return eta


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
    assert np.all(result >= 0)
    assert np.all(result <= 1.1)
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
    assert np.all(eta >= 0), np.min(eta)
    assert np.all(eta <= 1.1), np.max(eta)
    return eta


def eta_shap_rescale(weights, bias, baseline=None):
    # Try to replicate DeepLIFT (Rescale)

    n, m = weights.shape

    def f(w, b):
        return np.maximum(w + b, 0)

    dx = (weights - baseline).sum(0)
    baseline = baseline.sum(0)

    dy = 0.5 * (f(baseline + dx, bias) - f(baseline, bias))

    assert dy.shape == (m, ), dy.shape

    eta = np.divide(dy, dx, out=np.zeros_like(weights), where=dx!=0)
    assert eta.shape == (n, m), eta.shape
    assert np.all(eta >= 0)
    assert np.all(eta <= 1.01)
    return eta


def runner(b=10, n=4, m=100, dbias = 0, baselinew=1):
    x = 3*(np.random.random((b, n)) - 0.5) # (batch, #input)
    xb = baselinew*(np.random.random((b, n)) - 0.5) # (#input)
    #xb = x / 2.0
    w = 2*(np.random.random((n, m)) - 0.5) # (#input, #input2)
    bias = (np.random.random((m,)) - 0.5 + dbias)  # (#input2)
    #bias = 0 * np.ones_like(bias)
    #b = np.zeros_like(b)
    #x -= xb
    #xb = np.zeros_like(xb)
    # n = 5
    # x = np.array([-1.0] + [2/(n-1)]*(n-1))
    # w = np.array([1.0] * n)[...,np.newaxis]
    # b = np.array([0.0])
    # print(x.shape)
    # print(w.shape)
    # print(b.shape)
    #
    #
    #print("x: ", x)
    # print("b: ", b)
    session = tf.Session()
    games = np.expand_dims(x, -1) * w
    baseline = np.expand_dims(xb, -1) * w
    #print (baseline)

    assert games.shape == (b, n, m), games.shape

    #print("Games (players): ", games)
    #print("Outer (shape): ", outer.shape)
    #y = np.maximum(0, np.sum(games, 1)+b)
    #print("y: ", y)

    #print ("Shap Exact")
    eta_exact = eta_shap(games, bias=bias, baseline=baseline, method='exact').eval(session = session)
    #assert eta_exact.shape == games.shape, eta_exact.shape
    #print ("Eta shape:", eta_exact.shape)
    #print ('Eta exact:', eta_exact)
    #print ('Checksum: the following should be similar')
    #print (y - b)
    #np.sum(x * np.sum((np.expand_dims(w, 0) * eta_exact), -1), -1))

    #print ("Shap RevCanccel")
    eta_revcancel = eta_shap(games, bias=bias, baseline=baseline, method='revcancel').eval(session=session)
    assert eta_revcancel.shape == games.shape, eta_revcancel.shape

    eta_rescale = eta_shap(games, bias=bias, baseline=baseline, method='rescale').eval(session=session)
    assert eta_rescale.shape == games.shape, eta_rescale.shape

    #print ("Eta shape:", eta_revcancel.shape)
    #print ('Eta revcancel:', eta_revcancel)
    #print ('Checksum: the following should be similar')
    #print (y - np.maximum(0, b))
    #print (np.sum(x * np.sum((np.expand_dims(w, 0) * eta_revcancel), -1), -1))

    #print ("Shap Approx")
    eta_approx = eta_shap(games, bias=bias, baseline=baseline, method='approx').eval(session = session)
    assert eta_approx.shape == games.shape, eta_approx.shape

    #print ("Eta shape:", eta_approx.shape)
    #print ('Eta apprx:', eta_approx)
    #print ('Checksum: the following should be similar')
    #print (y - np.maximum(0, b))
    #print (np.sum(x * np.sum((np.expand_dims(w, 0) * eta_approx), -1), -1))

    #print ("Approx error")
    rmse_approx = np.sqrt(np.mean((eta_exact-eta_approx)**2))
    rmse_revcancel = np.sqrt(np.mean((eta_exact - eta_revcancel) ** 2))
    rmse_rescale = np.sqrt(np.mean((eta_exact - eta_rescale) ** 2))
    diff = np.sqrt(np.mean((eta_approx - eta_revcancel) ** 2))

    delta = games - baseline
    x = delta[0].flatten()
    y = (eta_exact[0] * delta[0]).flatten()

    # def f(x, a, b, c):
    #     return np.log(a**10 + c*np.exp(x-b)) - np.log(a**10+1)
    #
    # popt, pcov = scipy.optimize.curve_fit(f, x, y)
    # print (pcov)
    # plt.plot(np.sort(x), f(np.sort(x), *popt))

    # plt.figure()
    # plt.scatter(delta[0], eta_exact[0] * delta[0], label='exact')
    # plt.scatter(delta[0], eta_revcancel[0] * delta[0], label='rev_cancel')
    # plt.scatter(delta[0], eta_rescale[0] * delta[0], label='rescale')
    # plt.scatter(delta[0], eta_approx[0] * delta[0], label='approx')
    # plt.legend(loc='upper left')
    # plt.show()

    return np.array([rmse_approx, rmse_rescale, rmse_revcancel])
    #return np.array([diff, diff])


def test_time():
    import time
    times = []
    for _ in range(10):
        b = 100
        n = 10
        m = 100
        dbias = 0
        x = 3 * (np.random.random((b, n)) - 0.5)  # (batch, #input)
        xb = 1 * (np.random.random((b, n)) - 0.5)  # (#input)
        w = 2 * (np.random.random((n, m)) - 0.5)  # (#input, #input2)
        bias = 2 * (np.random.random((m,)) - 0.5 + dbias)  # (#input2)
        session = tf.Session()
        games = np.expand_dims(x, -1) * w
        baseline = np.expand_dims(xb, -1) * w

        t = time.process_time()
        eta_shap(games, bias=bias, baseline=baseline, method='approx').eval(session=session)
        times.append(time.process_time() - t)
    print (np.mean(times))


def test():
    #params = range(4, 20, 1)
    params = np.linspace(0, 4, 10)
    tests = 10
    result = np.zeros((len(params), tests, 3))
    for i, p in enumerate(params):
        for j in range(tests):
            result[i,j,:] = runner(b=1, n=18, m=1, dbias=0, baselinew=p)

    y = np.mean(result, 1)
    e = np.std(result, 1)

    plt.figure()
    for i in range(result.shape[-1]):
        plt.fill_between(params, y[:, i] + e[:, i] / 2, y[:, i] - e[:, i] / 2, alpha=.2)
        plt.plot(params, y[:, i])
    plt.legend(['Approx', 'Rescale', 'RevCancel'])
    plt.show()



if __name__ == "__main__":
    import numpy, time
    import numpy as np
    import matplotlib.pyplot as plt
    numpy.random.seed(int(time.time()))
    test()

