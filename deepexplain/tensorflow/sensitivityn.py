import numpy as np
import h5py
import math, os, operator, csv, pickle, sys
import plotly.offline as py
import plotly.graph_objs as go
from .stats import pearson_corr, kentall_tau, spearman_rho, mse, r2

CORRUPTION_MAX_RATIO = 1.0
REMOVE_VALUE = 0
RANDOM_TESTS = 3


def _ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def _run_model(x, y, model):
    p = model.predict(x, batch_size=128, verbose=0)
    p = np.sum(p*y, 1)
    return p  # size: (batch,)


def run_sensitivity_test(model, x, y, saliency_maps, names, task_name,
                   result_path='.', number_of_samples=np.inf):
    print('Running delta test...')
    sys.stdout.flush()
    assert len(saliency_maps) > 0
    maps = [np.copy(m) for m in saliency_maps]

    batch_size = len(x)
    map_shape = maps[0].shape
    saliency_length = np.prod(map_shape[1:])

    if len(x.shape) != len(map_shape):
        print ('Input shape (%s) is not equal to map shape (%s)' % (x.shape, map_shape))
        if len(x.shape) == len(map_shape) - 1 and map_shape[-1] == 1:
            print ('Trying to squeeze maps...')
            maps = [m.squeeze(-1) for m in maps]
            map_shape = maps[0].shape

    MAX_TESTS = 50
    feat_per_step = np.round(np.geomspace(1, saliency_length, MAX_TESTS)).astype(np.int32)
    feat_per_step = np.unique(feat_per_step)

    f = h5py.File(result_path + '/deltas.hdf5', 'a')
    if task_name in f:
        del f[task_name]
    T = f.create_group(task_name)
    T.attrs.create('testsize_nsamples', data=[batch_size, number_of_samples])
    T['sample_sizes'] = feat_per_step
    T.create_dataset('dy', shape=(len(feat_per_step), number_of_samples * batch_size))

    # For each n, sample number_of_samples subsets of n feature index
    masked_idx = []  # prepare space for storing sampled occlusion indexes
    for f_idx, f_step in enumerate(feat_per_step):
        masked_idx.append([np.random.choice(np.arange(saliency_length), f_step, replace=False)
                             for i in range(number_of_samples)])


    # Compute dy (actual delta output) for all
    y0 = _run_model(x, y, model) # run model on original data
    for i_nfeat, nfeat in enumerate(feat_per_step): # <-- given an n...
        dy = None
        for idxx in masked_idx[i_nfeat]: # <-- ..loop over all sampled subsets of cardinality n
            mask = np.ones(map_shape[1:]).flatten() # ie 32*32*1  (last dimension might have been reduced)
            mask[idxx] = 0
            mask = mask.reshape(map_shape[1:]) # now map is (1, 32, 32, 1)
            x_mod = x * mask
            dy_batch = _run_model(x_mod, y, model) - y0
            dy = np.vstack((dy, dy_batch)) if dy is not None else dy_batch

        # Store in H5
        T['dy'][i_nfeat, :] = dy.flatten()


    # Now run over the heatmaps, and get sum of masked pixels for each method
    for map, name in zip(maps, names):
        T.create_dataset(name, shape=(MAX_TESTS, number_of_samples*batch_size))

        for i_nfeat, nfeat in enumerate(feat_per_step):  # <-- given an n...
            dx = None
            for idxx in masked_idx[i_nfeat]: # <-- ..loop over all sampled subsets of cardinality n
                mask = np.zeros(map_shape[1:]).flatten()  # ie 32*32*1
                mask[idxx] = 1
                mask = mask.reshape(map_shape[1:])
                map_mod = map * mask
                masked_mask_sum = np.sum(map_mod.reshape(batch_size, -1), 1)
                dx = np.vstack((dx, masked_mask_sum)) if dx is not None else masked_mask_sum
            # Store in H5
            T[name][i_nfeat, :] = -1.0 * dx.flatten()

    f.flush()
    f.close()
    print('Done\n\n\n')

