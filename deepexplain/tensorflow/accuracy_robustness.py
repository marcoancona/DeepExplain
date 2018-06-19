import numpy as np
import h5py
import math, os, operator, csv, pickle
import plotly.offline as py
import plotly.graph_objs as go
from PIL import Image
import scipy.misc

CORRUPTION_MAX_RATIO = 1
REMOVE_VALUE = 0
RANDOM_TESTS = 3

SAVE_IMAGES = 0
path = None

def _ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def _run_model(x, y, model, mode):
    if mode == 'accuracy':
        return model.evaluate(x, y, batch_size=10, verbose=0)[1]
    elif mode == 'prediction':
        p = model.predict(x, batch_size=10, verbose=0)
        p = np.sum(p*y, 1)
        return np.mean(p)
    else:
        raise RuntimeError('Mode non valid')


def _corrupt_loop(eval_x, eval_y, rank, model, feat_per_step, mode, method_name=None):
    x_neg = REMOVE_VALUE
    x = eval_x
    input_shape = eval_x.shape
    batch_size = input_shape[0]

    map_size = rank.shape
    features_count = np.prod(map_size[1:])
    metric = [_run_model(x, eval_y, model, mode=mode)]
    rank_flatten = rank.reshape((batch_size, -1))

    steps = int(features_count * CORRUPTION_MAX_RATIO / feat_per_step)

    for i in range(steps):
        batch_mask = np.zeros_like(rank_flatten, dtype=np.bool)
        batch_mask[[(rank_flatten >= i * feat_per_step) * (rank_flatten < (i+1)*feat_per_step)]] = True
        batch_mask = batch_mask.reshape(map_size)
        x = ~batch_mask * x + batch_mask * x_neg
        if i < SAVE_IMAGES:
            scipy.misc.imsave(path+'/img_%s_%3d.png' % (method_name if method_name is not None else '', i), x[0])
            #Image.fromarray(x[0]).save(path+'/img_%s_%3d.png' % (method_name if method_name is not None else '', i))
        metric.append(_run_model(x, eval_y, model, mode=mode))
    return metric


def run_robustness_test(model, x, y, original_maps, names, task_name, feat_per_step,
                        result_path='.', mode='accuracy', reduce_dim=None):
    print('Running robustness test...')
    assert len(original_maps) > 0
    maps = [np.copy(m) for m in original_maps]

    global path
    path = result_path + '/' + task_name
    _ensure_dir(path)

    batch_size = len(x)
    map_shape = maps[0].shape
    saliency_length = np.prod(map_shape[1:])

    # do reduce_sum in the beginning
    if reduce_dim is not None:
        print ('--> using reduce_sum of dimension {} of heatmap'.format(reduce_dim))
        maps = [np.sum(m, reduce_dim, keepdims=True) for m in maps]
        map_shape = maps[0].shape
        saliency_length = np.prod(map_shape[1:])

    if len(x.shape) != len(map_shape):
        print ('Input shape (%s) is not equal to map shape (%s)' % (x.shape, map_shape))
        if len(x.shape) == len(map_shape) - 1 and map_shape[-1] == 1:
            print ('Trying to squeeze maps...')
            maps = [m.squeeze(-1) for m in maps]
            map_shape = maps[0].shape


    plot_data_max = {}
    colors_max = ['#ff0000',
                 '#772014',
                 '#3f220f',
                 '#19180a',
                 '#5f0f40',
                 '#321325',
                 '#3e1591',
                 '#e5ac10']
    plot_data_min = {}
    colors_min = ['#1b98e0',
                 '#0066d0',
                 '#006494',
                 '#19180a',
                 '#13293d',
                 '#6170c6',
                 '#baffda',
                 '#99e3ff']
    plot_data_other = {}

    for map, name in zip(maps, names):
        map = map.reshape(map_shape)
        map_flat = map.reshape(batch_size, -1)
        rank_max = np.argsort(np.argsort(map_flat * -1.0, axis=1), axis=1).reshape(map_shape)
        rank_min = np.argsort(np.argsort(map_flat, axis=1), axis=1).reshape(map_shape)
        plot_data_max[name] = _corrupt_loop(x, y, rank_max, model, feat_per_step, mode=mode, method_name=name)
        plot_data_min[name] = _corrupt_loop(x, y, rank_min, model, feat_per_step, mode=mode, method_name=name)

    # Do random
    rand_results = []
    for j in range(RANDOM_TESTS):
        rank_rand = np.stack([np.random.permutation(np.arange(0, saliency_length)) for i in range(len(x))]).reshape(map_shape)
        rand_results.append(_corrupt_loop(x, y, rank_rand, model, feat_per_step, mode=mode, method_name='rand'))
    plot_data_other['rand'] = np.mean(rand_results, 0)
    plot_data_other['_rand_std'] = np.std(rand_results, 0)

    n_ticks = len(plot_data_other['rand'])
    x_ticks = np.linspace(0, 100, n_ticks)

    traces = []
    for idx, map_name in enumerate(plot_data_max):
        traces.append(go.Scatter(
            x=x_ticks,
            y=plot_data_max[map_name],
            name=map_name+'+',
            marker=dict(color=colors_max[idx % len(colors_max)]),
            yaxis='y',
            xaxis='x'
        ))

    for idx, map_name in enumerate(plot_data_min):
        traces.append(go.Scatter(
            x=x_ticks,
            y=plot_data_min[map_name],
            name=map_name + '-',
            marker=dict(color=colors_min[idx % len(colors_min)]),
            yaxis='y',
            xaxis='x'
        ))

    for map_name in plot_data_other:
        if map_name[0] != '_':
            traces.append(go.Scatter(
                x=x_ticks,
                y=plot_data_other[map_name],
                name=map_name,
                marker=dict(color='green'),
                yaxis='y',
                xaxis='x'
            ))

    layout = go.Layout(
        title='%s vs perturbation rate' % mode.capitalize(),
        hovermode='closest',
        xaxis=dict(
            title='Percent perturbed ',
            ticklen=5,
            zeroline=False,
            gridwidth=2,
        ),
        yaxis=dict(
            title=mode,
            ticklen=5,
            gridwidth=2,
            domain=[0.25, 0.75]
        )
    )


    figure = go.Figure(data=traces, layout=layout)
    py.plot(figure, filename=path+'/robustness_'+mode+'.html', auto_open=False)

    print('Storing robustness results...')
    data = {}
    for idx, map_name in enumerate(plot_data_min):
        data[map_name+'-'] = plot_data_min[map_name]
    for map_name in plot_data_other:
        data[map_name] = plot_data_other[map_name]
    for idx, map_name in enumerate(plot_data_max):
        data[map_name+'+'] = plot_data_max[map_name]
    fileHandle = open(path+'/robustness_'+mode+'.p', "wb")
    pickle.dump(data, fileHandle)


    # f = h5py.File(result_path+'/all_heatmaps.hdf5', 'a')
    # T = f[task_name]
    # if mode in T.keys():
    #     del T[mode]
    # R = T.create_group(mode)
    # for map_name in plot_data_max:
    #     R.create_dataset(map_name+'+', data=plot_data_max[map_name])
    # for map_name in plot_data_min:
    #     R.create_dataset(map_name+'-', data=plot_data_min[map_name])
    # for map_name in plot_data_other:
    #     R.create_dataset(map_name, data=plot_data_other[map_name])
    # R.create_dataset('x_ticks', data=x_ticks)
    # f.flush()
    # f.close()

    # Integrate area
    areas = {}
    if mode == 'accuracy':
        for idx, map_name in enumerate(plot_data_min):
            acc_plus = np.array(plot_data_max[map_name])
            acc_minus = np.array(plot_data_min[map_name])
            rand = np.array(plot_data_other['rand'])
            area_plus = np.sum(rand - acc_plus) / len(rand) * 100
            area_min = np.sum(acc_minus - rand) / len(rand) * 100
            area = area_min + area_plus
            areas[map_name] = (area_plus, area_min, area)
            with open(path+'/results.csv', 'w') as csvfile:
                for n in sorted(areas, key=lambda x: areas[x][2], reverse=True):
                    csvfile.write('%s\t%.2f\t%.2f\t%.2f\n' % (n, areas[n][0], areas[n][1], areas[n][2]))
            csvfile.close()

    print('Done')

