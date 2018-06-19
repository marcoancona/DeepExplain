import numpy as np
from scipy import stats
from sklearn.metrics import r2_score


def pearson_corr(y, x):
    """
    Calculates Pearson correlation (https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)
    :param y: signal 1
    :param x: signal 2
    :return: number in [-1, 1]
    """
    n = len(x)
    x_sum = np.sum(x)
    y_sum = np.sum(y)
    xy_sum = np.sum(x * y)
    x2_sum = np.sum(x * x)
    y2_sum = np.sum(y * y)

    r_num = n * xy_sum - x_sum * y_sum
    r_den_x = np.sqrt(n * x2_sum - x_sum * x_sum)
    r_den_y = np.sqrt(n * y2_sum - y_sum * y_sum)
    r = r_num / (r_den_x * r_den_y)
    return r


def kentall_tau(y, x):
    tau, p_value = stats.kendalltau(x, y)
    return tau

def spearman_rho(y, x):
    rho, p_value = stats.spearmanr(x, y)
    return rho

def mse(y, x):
    return np.mean((x-y) ** 2)


def r2(y, x):
    # Assumes a linear fit y = x
    return r2_score(y, x)