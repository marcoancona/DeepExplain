import numpy as np
import itertools
from itertools import chain, combinations
import scipy.special
fact = scipy.special.factorial

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
    y = np.sum(x*w, -1) + b
    return np.maximum(0, y)


def powerset(iterable):
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    xs = list(iterable)
    # note we return an iterator rather than a list
    return chain.from_iterable(combinations(xs, n) for n in range(len(xs) + 1))

def vec_bin_array(arr, m):
    """
    Arguments:
    arr: Numpy array of positive integers
    m: Number of bits of each integer to retain

    Returns a copy of arr with every element replaced with a bit vector.
    Bits encoded as int8's.
    """
    to_str_func = np.vectorize(lambda x: np.binary_repr(x).zfill(m))
    strs = to_str_func(arr)
    ret = np.zeros(list(arr.shape) + [m], dtype=np.int8)
    for bit_ix in range(0, m):
        fetch_bit_func = np.vectorize(lambda x: x[bit_ix] == '1')
        ret[...,bit_ix] = fetch_bit_func(strs).astype("int8")

    return ret


def compute_shapley(inputs, f, baseline=None):
    print ("Exact Shapley (v2)")
    print (inputs.shape)
    if baseline is None:
        baseline = np.zeros_like(inputs)
    results = np.zeros(inputs.shape)
    n = len(inputs)
    # Create powerset binary mask with shape (2**n, n)
    # Note: we first exclude column with index index, then we add it
    mask = vec_bin_array(np.arange(2 ** (n-1)), n-1)
    # assert mask.shape == (2**(n-1), n-1), 'Mask shape does not match'
    coeff = (fact(mask.sum(1)) * fact(n - mask.sum(1) - 1)) / fact(n)

    for index in range(n):
        # Copy mask and set the current player active
        mask_wo_index = np.insert(mask, index, np.zeros(2 ** (n-1)), axis=1)
        mask_wi_index = np.insert(mask, index, np.ones(2 ** (n-1)), axis=1)
        # print(mask_wo_index.shape)
        # assert mask_wo_index.shape == (2 ** (n - 1), n), 'Mask shape does not match'

        run_wo_i = f(inputs * mask_wo_index)  # run all masks at once
        run_wi_i = f(inputs * mask_wi_index)  # run all masks at once
        # assert len(run_wi_i.shape) == 1, 'Result shape len does not match %s' % (run_wi_i.shape,)
        r = (run_wi_i - run_wo_i) * coeff
        results[index] = r.sum()

    return results


def compute_shapley_l(inputs, f, baseline=None):
    print ("Exact Shapley")
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
    n = 5
    x = np.array([-1.0] + [2/(n-1)]*(n-1))
    w = np.array([1.0] * n)
    b = 0

    print ("x: ", x)
    print ("b: ", b)
    print ("x * w: ", x*w)
    print ("y: ", f_linear_relu(x, w, b))
    shapley = compute_shapley(x, lambda x: f_linear_relu(x, w, b))
    print ("Shapley: ", shapley)
    print ("Sum: ", np.sum(shapley))



if __name__ == "__main__":
    main()