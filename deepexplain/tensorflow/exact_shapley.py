import numpy as np
import itertools
from itertools import chain, combinations
from math import factorial as fact


# def shapley_max(x, r):
#     # sort so that at each step we know either the input value
#     # or the reference value of the next feature will be the next largest value
#
#     perm = sortperm(collect(zip(x, r)), by=maximum, rev=true)
#     xsorted = x[perm]
#     rsorted = r[perm]
#     M = length(x)
#     path = zeros(M)
#     weight = 1.0
#     num_ones = 0
#     phi = zeros(M)
#     last_val = -Inf
#     weight_scale = 1.0
#
#     for i in range(1, M):
#
#         largest_remaining = -Inf if i == M else max(xsorted[i + 1], rsorted[i + 1])
#         if xsorted[i] >= largest_remaining:
#
#             path[i] = 1
#
#             for j in range(1, i):
#                 if path[j] == 1:
#                     phi[perm[j]] += max(last_val, xsorted[i]) * weight * ((num_ones + 1) / i) / (num_ones + 1)
#                 else:
#                     phi[perm[j]] -= max(last_val, xsorted[i]) * weight * ((num_ones + 1) / i) / (i - num_ones - 1)
#
#             path[i] = 0
#             weight_scale = (i - num_ones) / i
#
#         if rsorted[i] >= largest_remaining:
#
#             path[i] = 0
#             for j in range(1, i):
#                 if path[j] == 1:
#                     phi[perm[j]] += max(last_val, rsorted[i]) * weight * ((i - num_ones) / i) / num_ones
#                 else:
#                     phi[perm[j]] -= max(last_val, rsorted[i]) * weight * ((i - num_ones) / i) / (i - num_ones)
#             path[i] = 1
#             num_ones += 1
#             weight_scale = num_ones / i
#
#         if xsorted[i] >= largest_remaining and rsorted[i] >= largest_remaining:
#             break
#
#         last_val = max(min(xsorted[i], rsorted[i]), last_val)
#         weight *= weight_scale
#     return phi


def f_max(inputs):
    return np.max(inputs)


def f_linear_relu(x, w, b):
    y = np.sum(x*w) + b
    return max(0, y)


def powerset(iterable):
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    xs = list(iterable)
    # note we return an iterator rather than a list
    return chain.from_iterable(combinations(xs, n) for n in range(len(xs) + 1))


def compute_shapley(inputs, f, baseline=None):
    if baseline is None:
        baseline = np.zeros_like(inputs)
    results = np.zeros(inputs.shape)
    for index in range(len(inputs)):
        # Find indices of all players except 'index'
        players = set(list(range(len(inputs))))
        players.remove(index)

        r = 0.0
        for S in powerset(players):
            S_with_i = set(S)
            S_with_i.add(index)
            I_with_i = np.array(baseline)
            I_with_i[list(S_with_i)] = inputs[list(S_with_i)]
            I = np.array(baseline)
            I[list(S)] = inputs[list(S)]
            r += (fact(len(S)) * fact(len(inputs) - len(S) - 1) / fact(len(inputs))) * (f(I_with_i) - f(I))
        results[index] = r
    return results


def main():
    x = np.random.rand(5)
    w = 5 * np.random.rand(5)
    b = 10 * np.random.rand(1)
    b = -3
    print ("x: ", x)
    print ("b: ", b)
    print ("x * w: ", x*w)
    print ("y: ", f_linear_relu(x, w, b))
    shapley = compute_shapley(x, lambda x: f_linear_relu(x, w, b))
    print ("Shapley: ", shapley)
    print ("Sum: ", np.sum(shapley))



if __name__ == "__main__":
    main()