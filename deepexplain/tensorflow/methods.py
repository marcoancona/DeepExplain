from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
from skimage.util import view_as_windows
import warnings
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import nn_grad, math_grad
from collections import OrderedDict
from .deep_shapley import eta_shap

SUPPORTED_ACTIVATIONS = [
    'Relu', 'Elu', 'Sigmoid', 'Tanh', 'Softplus'
]

UNSUPPORTED_ACTIVATIONS = [
    'CRelu', 'Relu6', 'Softsign'
]

_ENABLED_METHOD_CLASS = None
_GRAD_OVERRIDE_CHECKFLAG = 0
_MATMUL_GRAD_OVERRIDE_CHECKFLAG = 0

SESSION = None


# -----------------------------------------------------------------------------
# UTILITY FUNCTIONS
# -----------------------------------------------------------------------------


def activation(type):
    """
    Returns Tensorflow's activation op, given its type
    :param type: string
    :return: op
    """
    if type not in SUPPORTED_ACTIVATIONS:
        warnings.warn('Activation function (%s) not supported' % type)
    f = getattr(tf.nn, type.lower())
    return f


def original_grad(op, grad):
    """
    Return original Tensorflow gradient for an op
    :param op: op
    :param grad: Tensor
    :return: Tensor
    """
    #if op.type not in SUPPORTED_ACTIVATIONS:
    #    warnings.warn('Activation function (%s) not supported' % op.type)
    opname = '_%sGrad' % op.type
    if hasattr(nn_grad, opname):
        f = getattr(nn_grad, opname)
    else:
        f = getattr(math_grad, opname)
    return f(op, grad)


# -----------------------------------------------------------------------------
# ATTRIBUTION METHODS BASE CLASSES
# -----------------------------------------------------------------------------


class AttributionMethod(object):
    """
    Attribution method base class
    """
    def __init__(self, T, X, xs, session, keras_learning_phase=None):
        self.T = T
        self.X = X
        self.xs = xs
        self.session = session
        self.keras_learning_phase = keras_learning_phase
        self.has_multiple_inputs = type(self.X) is list or type(self.X) is tuple
        print ('Model with multiple inputs: ', self.has_multiple_inputs)

    def session_run(self, T, xs):
        feed_dict = {}
        if self.has_multiple_inputs:
            if len(xs) != len(self.X):
                raise RuntimeError('List of input tensors and input data have different lengths (%s and %s)'
                                   % (str(len(xs)), str(len(self.X))))
            for k, v in zip(self.X, xs):
                feed_dict[k] = v
        else:
            feed_dict[self.X] = xs

        if self.keras_learning_phase is not None:
            feed_dict[self.keras_learning_phase] = 0
        return self.session.run(T, feed_dict)

    def _set_check_baseline(self):
        if self.baseline is None:
            if self.has_multiple_inputs:
                self.baseline = [np.zeros((1,) + xi.shape[1:]) for xi in self.xs]
            else:
                self.baseline = np.zeros((1,) + self.xs.shape[1:])

        else:
            if self.has_multiple_inputs:
                for i, xi in enumerate(self.xs):
                    if self.baseline[i].shape == self.xs[i].shape[1:]:
                        self.baseline[i] = np.expand_dims(self.baseline[i], 0)
                    else:
                        raise RuntimeError('Baseline shape %s does not match expected shape %s'
                                           % (self.baseline[i].shape, self.xs[i].shape[1:]))
            else:
                if self.baseline.shape == self.xs.shape[1:]:
                    self.baseline = np.expand_dims(self.baseline, 0)
                else:
                    raise RuntimeError('Baseline shape %s does not match expected shape %s'
                                       % (self.baseline.shape, self.xs.shape[1:]))


class GradientBasedMethod(AttributionMethod):
    """
    Base class for gradient-based attribution methods
    """
    def get_symbolic_attribution(self):
        return tf.gradients(self.T, self.X)

    def run(self):
        attributions = self.get_symbolic_attribution()
        results =  self.session_run(attributions, self.xs)
        return results[0] if not self.has_multiple_inputs else results

    @classmethod
    def nonlinearity_grad_override(cls, op, grad):
        return original_grad(op, grad)

    @classmethod
    def matmul_grad_override(cls, op, grad):
        return original_grad(op, grad)

    @classmethod
    def convolution_grad_override(cls, op, grad):
        print ('Conv')
        print (op)
        print (grad)
        print (original_grad(op, grad))
        return original_grad(op, grad)

    @classmethod
    def maxpool_grad_override(cls, op, grad):
        print('maxpool')
        print(op)
        print(grad)
        print(original_grad(op, grad))
        return original_grad(op, grad)




class PerturbationBasedMethod(AttributionMethod):
    """
       Base class for perturbation-based attribution methods
       """
    def __init__(self, T, X, xs, session, keras_learning_phase):
        super(PerturbationBasedMethod, self).__init__(T, X, xs, session, keras_learning_phase)
        self.base_activation = None

    def _run_input(self, x):
        return self.session_run(self.T, x)

    def _run_original(self):
        return self._run_input(self.xs)

    def run(self):
        raise RuntimeError('Abstract: cannot run PerturbationBasedMethod')


# -----------------------------------------------------------------------------
# ATTRIBUTION METHODS
# -----------------------------------------------------------------------------
"""
Returns zero attributions. For testing only.
"""


class DummyZero(GradientBasedMethod):

    def get_symbolic_attribution(self,):
        return tf.gradients(self.T, self.X)

    @classmethod
    def nonlinearity_grad_override(cls, op, grad):
        input = op.inputs[0]
        return tf.zeros_like(input)

"""
Saliency maps
https://arxiv.org/abs/1312.6034
"""


class Saliency(GradientBasedMethod):

    def get_symbolic_attribution(self):
        return [tf.abs(g) for g in tf.gradients(self.T, self.X)]


"""
Gradient * Input
https://arxiv.org/pdf/1704.02685.pdf - https://arxiv.org/abs/1611.07270
"""


class GradientXInput(GradientBasedMethod):

    def get_symbolic_attribution(self):
        return [g * x for g, x in zip(
            tf.gradients(self.T, self.X),
            self.X if self.has_multiple_inputs else [self.X])]


"""
Integrated Gradients
https://arxiv.org/pdf/1703.01365.pdf
"""


class IntegratedGradients(GradientBasedMethod):

    def __init__(self, T, X, xs, session, keras_learning_phase, steps=100, baseline=None):
        super(IntegratedGradients, self).__init__(T, X, xs, session, keras_learning_phase)
        self.steps = steps
        self.baseline = baseline

    def run(self):
        # Check user baseline or set default one
        self._set_check_baseline()

        attributions = self.get_symbolic_attribution()
        gradient = None
        for alpha in list(np.linspace(1. / self.steps, 1.0, self.steps)):
            xs_mod = [xs * alpha for xs in self.xs] if self.has_multiple_inputs else self.xs * alpha
            _attr = self.session_run(attributions, xs_mod)
            if gradient is None: gradient = _attr
            else: gradient = [g + a for g, a in zip(gradient, _attr)]

        results = [g * (x - b) / self.steps for g, x, b in zip(
            gradient,
            self.xs if self.has_multiple_inputs else [self.xs],
            self.baseline if self.has_multiple_inputs else [self.baseline])]

        return results[0] if not self.has_multiple_inputs else results


"""
Layer-wise Relevance Propagation with epsilon rule
http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140
"""


class EpsilonLRP(GradientBasedMethod):
    eps = None

    def __init__(self, T, X, xs, session, keras_learning_phase, epsilon=1e-4):
        super(EpsilonLRP, self).__init__(T, X, xs, session, keras_learning_phase)
        assert epsilon > 0.0, 'LRP epsilon must be greater than zero'
        global eps
        eps = epsilon

    def get_symbolic_attribution(self):
        return [g * x for g, x in zip(
            tf.gradients(self.T, self.X),
            self.X if self.has_multiple_inputs else [self.X])]

    @classmethod
    def nonlinearity_grad_override(cls, op, grad):
        output = op.outputs[0]
        input = op.inputs[0]
        return grad * output / (input + eps *
                                tf.where(input >= 0, tf.ones_like(input), -1 * tf.ones_like(input)))

"""
DeepLIFT
This reformulation only considers the "Rescale" rule
https://arxiv.org/abs/1704.02685
"""


class DeepLIFTRescale(GradientBasedMethod):

    _deeplift_ref = {}

    def __init__(self, T, X, xs, session, keras_learning_phase, baseline=None):
        super(DeepLIFTRescale, self).__init__(T, X, xs, session, keras_learning_phase)
        self.baseline = baseline

    def get_symbolic_attribution(self):
        return [g * (x - b) for g, x, b in zip(
            tf.gradients(self.T, self.X),
            self.X if self.has_multiple_inputs else [self.X],
            self.baseline if self.has_multiple_inputs else [self.baseline])]

    @classmethod
    def nonlinearity_grad_override(cls, op, grad):
        output = op.outputs[0]
        input = op.inputs[0]
        ref_input = cls._deeplift_ref[op.name]
        ref_output = activation(op.type)(ref_input)
        delta_out = output - ref_output
        delta_in = input - ref_input
        instant_grad = activation(op.type)(0.5 * (ref_input + input))
        return tf.where(tf.abs(delta_in) > 1e-5, grad * delta_out / delta_in,
                               original_grad(instant_grad.op, grad))

    def run(self):
        # Check user baseline or set default one
        self._set_check_baseline()

        # Init references with a forward pass
        self._init_references()

        # Run the default run
        return super(DeepLIFTRescale, self).run()

    def _init_references(self):
        print ('DeepLIFT: computing references...')
        sys.stdout.flush()
        self._deeplift_ref.clear()
        ops = []
        g = tf.get_default_graph()
        for op in g.get_operations():
            if len(op.inputs) > 0 and not op.name.startswith('gradients'):
                if op.type in SUPPORTED_ACTIVATIONS:
                    ops.append(op)
        print (ops)
        print (self.baseline)
        YR = self.session_run([o.inputs[0] for o in ops], self.baseline)
        for (r, op) in zip(YR, ops):
            self._deeplift_ref[op.name] = r
        print (self._deeplift_ref)
        print('DeepLIFT: references ready')
        sys.stdout.flush()


"""
Linear
"""


class Linear(GradientBasedMethod):
    def __init__(self, T, X, xs, session, keras_learning_phase):
        super(Linear, self).__init__(T, X, xs, session, keras_learning_phase)

    def get_symbolic_attribution(self):
        return [g * x for g, x in zip(
            tf.gradients(self.T, self.X),
            self.X if self.has_multiple_inputs else [self.X])]

    @classmethod
    def nonlinearity_grad_override(cls, op, grad):
        return grad


"""
DeepShapley
"""


class DeepShapley(GradientBasedMethod):

    _deepshap_ref = {}
    _deepshap_for = {}

    def __init__(self, T, X, xs, session, keras_learning_phase, baseline=None):
        super(DeepShapley, self).__init__(T, X, xs, session, keras_learning_phase)
        self.baseline = baseline

    def get_symbolic_attribution(self):
        return [g * (x - b) for g, x, b in zip(
            tf.gradients(self.T, self.X),
            self.X if self.has_multiple_inputs else [self.X],
            self.baseline if self.has_multiple_inputs else [self.baseline])]

    @classmethod
    def convolution_grad_override(cls, op, grad):
        #We assume this matmul is followed by a BiasAdd and a Relu op
        players = cls._deepshap_for[op.name + "_x"]
        kernel = cls._deepshap_for[op.name + "_w"]
        bias = cls._deepshap_for[op.name + '_b']
        reference = cls._deepshap_ref[op.name + "_x"]
        print (op)
        # print (players.shape)
        # print (weights.shape)
        # print (bias.shape)
        # print (reference.shape)
        print ('Conv2d override: ', op.name)
        grad_shape = players.shape

        g1, g2 = original_grad(op, grad)

        # Convert Conv2D into MatMul operation and proceed
        ksizes = (1,) + kernel.shape[:-1]
        strides = op.get_attr('strides')
        padding = op.get_attr('padding')
        rates = op.get_attr('dilations')

        def extract_patches(x):
            return tf.extract_image_patches(
                x,
                ksizes=ksizes,
                strides=strides,
                rates=rates,
                padding=padding
            )

        def extract_patches_inverse(x, y):
            _x = tf.zeros_like(x)
            print (_x)
            _y = extract_patches(_x)
            print (_y)
            y = tf.check_numerics(
                y,
                'y contains nans',
            )
            grad = tf.gradients(_y, _x)[0]
            # Divide by grad, to "average" together the overlapping patches
            # otherwise they would simply sum up
            return tf.gradients(_y, _x, grad_ys=y)[0] / grad


        # Extract patches following same settings of convolution
        patches = extract_patches(players)
        reference = extract_patches(reference)

        # Reshape patches to have all elements involved in convolution together in the last dimension
        _players = tf.reshape(patches, (-1, np.prod(ksizes)))
        reference = tf.reshape(reference, (-1, np.prod(ksizes)))
        # Do the same for the kernel, except that we aggregate all kernel values in the first dimension
        weights = kernel.reshape(-1, kernel.shape[-1])

        _players = _players.eval(session=SESSION)
        reference = reference.eval(session=SESSION)

        print ("Players", _players.shape)
        print ("Reference", reference.shape)
        print ("Kernel", weights.shape)
        print ("Bias", bias.shape)

        grad = tf.reshape(grad, (-1, kernel.shape[-1]))
        print ("Grad", grad.shape)


        grad_list = []
        for idx in range(_players.shape[0]):
            outer = np.expand_dims(_players[idx], 1) * weights
            outer_b = np.expand_dims(reference[idx % reference.shape[0]], 1) * weights
            eta = eta_shap(outer, bias, outer_b)
            #print ("Eta", eta.shape)
            grad_list.append(tf.squeeze(tf.matmul(tf.expand_dims(grad[idx], 0), weights * eta, transpose_b=True), axis=0))

        result = tf.stack(grad_list)
        print ("Result,prereshape", result.shape)
        result = extract_patches_inverse(players, result)
        print("Result", result.shape)

        #assert result.get_shape().as_list()[1:] == g1.get_shape().as_list()[1:], \
            #"Gradient got shape %s, while expecting %s" % (result.get_shape().as_list(), g1.get_shape().as_list())
        print ('Return')
        return result, g2

    @classmethod
    def maxpool_grad_override(cls, op, grad):
        players = cls._deepshap_for[op.name + "_x"]
        reference = cls._deepshap_ref[op.name + "_x"]

        b, w, h, c = players.shape
        _, kw, kh, _ = op.get_attr('ksize')
        hw, hh = w // kw, h // kh
        pad = [[0, 0], [0, 0]]
        x = tf.space_to_batch_nd(players, [kw, kh], pad)
        r = tf.space_to_batch_nd(reference, [kw, kh], pad)
        x = tf.reshape(x, (kw * kh, -1))
        r = tf.reshape(r, (kw * kh, -1))
        grad_flat = tf.reshape(grad, (b, -1))

        x_np = tf.transpose(x, (1, 0)).eval(session=SESSION)
        r_np = tf.transpose(r, (1, 0)).eval(session=SESSION)
        grad_list = []

        print ("x_rp", x_np.shape)
        print ("r_rp", r_np.shape)

        result = eta_shap(np.expand_dims(x_np, 1),
                          baseline=np.expand_dims(np.repeat(r_np, b, 0), -1),
                          weights=grad_flat,
                          method='exact',
                          fun = lambda x: np.max(x, 1))
        # for idx in range(x_np.shape[0]):
        #     players = np.expand_dims(x_np[idx], 1)
        #     baseline = np.expand_dims(r_np[0], 1)  # Baseline always the same
        #     # if (np.count_nonzero(players-baseline)) == 0:
        #     #     grad_list.append(np.squeeze(players))
        #     #     continue
        #     eta = eta_shap(players, None, baseline, method='exact', fun=lambda x: np.max(x, 1))
        #     grad_list.append(grad_flat[idx] * np.squeeze(eta))
        #result = tf.stack(grad_list)

        # Original gradient of maxpool (for sanity test)
        # Uncomment the following lines to discard Shapley and use a custom implementation of max pooling
        # argmax_ = tf.argmax(x_np, 1)
        # result = tf.one_hot(argmax_, kw * kh, axis=1) * tf.expand_dims(grad_flat, -1)
        # end override

        print ("result", result.shape)
        result = tf.transpose(result, (1, 0))
        result = tf.reshape(result, (kw * kh, b, -1))
        result = tf.reshape(result, (-1, hw, hh, c))
        result = tf.batch_to_space_nd(result, [kw, kh], pad)

        original = original_grad(op, grad)

        return result

    @classmethod
    def nonlinearity_grad_override(cls, op, grad):
        #output = op.outputs[0]
        # input = op.inputs[0]
        # Identify function
        return grad
        #return tf.where(input > 0, grad, 0.3*grad)

    @classmethod
    def matmul_grad_override(cls, op, grad):
        #We assume this matmul is followed by a BiasAdd and a Relu op
        players = cls._deepshap_for[op.name + "_x"]
        weights = cls._deepshap_for[op.name + "_w"]
        bias = cls._deepshap_for[op.name + '_b']
        reference = cls._deepshap_ref[op.name + "_x"]

        print ("Players", players.shape)
        print ("Weights", weights.shape)
        # print (bias.shape)
        print ("Reference", reference.shape)
        print ('Matmul override: ', op.name)

        g1, g2 = original_grad(op, grad)


        #return g1, g2
        if 'dense_3' in op.name:
            print ("\t skipping...")
            return g1, g2

        result = eta_shap(np.expand_dims(players, -1) * np.expand_dims(weights, 0),
                          baseline=np.expand_dims(reference[0], 1) * weights,
                          bias=bias,
                          weights=grad,
                          method='approx',
                          )

        # grad_list = []
        # for idx in range(players.shape[0]):
        #     outer = np.expand_dims(players[idx], 1) * weights
        #     outer_b = np.expand_dims(reference[0], 1) * weights
        #     eta = eta_shap(outer, bias, outer_b)
        #     grad_list.append(tf.squeeze(tf.matmul(tf.expand_dims(grad[idx], 0), weights * eta, transpose_b=True), axis=0))
        # result = tf.stack(grad_list)

        assert result.get_shape().as_list()[1:] == g1.get_shape().as_list()[1:], \
            "Gradient got shape %s, while expecting %s" % (result.get_shape().as_list(), g1.get_shape().as_list())
        print ('Return')
        return result, g2

    def run(self):
        # Check user baseline or set default one
        self._set_check_baseline()

        # Init references with a forward pass
        self._init_references()

        # Run the default run
        return super(DeepShapley, self).run()

    def _init_references(self):
        print ('Shapley: computing references...')
        sys.stdout.flush()
        self._deepshap_ref.clear()
        ops = []
        tensors = []
        g = tf.get_default_graph()
        for op in g.get_operations():
            if len(op.inputs) > 0 and not 'gradients' in op.name and 'model' in op.name:
                print (op.type + ": " + op.name)
                if op.type == 'MatMul':
                    ops.append(op.name + "_x")
                    tensors.append(op.inputs[0])
                    ops.append(op.name + "_w")
                    tensors.append(op.inputs[1])
                elif op.type == 'BiasAdd':
                    ops.append(op.name[:-7] + "MatMul_b")
                    tensors.append(op.inputs[1])
                    ops.append(op.name[:-7] + "convolution_b")
                    tensors.append(op.inputs[1])
                elif op.type == 'MaxPool':
                    ops.append(op.name + "_x")
                    tensors.append(op.inputs[0])
                elif op.type == 'Conv2D':
                    ops.append(op.name + "_x")
                    tensors.append(op.inputs[0])
                    ops.append(op.name + "_w")
                    tensors.append(op.inputs[1])

        YXS = self.session_run(tensors, self.xs)
        YR = self.session_run(tensors, self.baseline)
        for (r, opName) in zip(YR, ops):
            self._deepshap_ref[opName] = r
        for (r, opName) in zip(YXS, ops):
            self._deepshap_for[opName] = r
        for k in self._deepshap_ref.keys():
            print (k, self._deepshap_ref[k].shape)
        print('Shapley: references ready')
        sys.stdout.flush()


"""
Occlusion method
Generalization of the grey-box method presented in https://arxiv.org/pdf/1311.2901.pdf
This method performs a systematic perturbation of contiguous hyperpatches in the input,
replacing each patch with a user-defined value (by default 0).

window_shape : integer or tuple of length xs_ndim
Defines the shape of the elementary n-dimensional orthotope the rolling window view.
If an integer is given, the shape will be a hypercube of sidelength given by its value.

step : integer or tuple of length xs_ndim
Indicates step size at which extraction shall be performed.
If integer is given, then the step is uniform in all dimensions.
"""


class Occlusion(PerturbationBasedMethod):

    def __init__(self, T, X, xs, session, keras_learning_phase, window_shape=None, step=None):
        super(Occlusion, self).__init__(T, X, xs, session, keras_learning_phase)
        if self.has_multiple_inputs:
            raise RuntimeError('Multiple inputs not yet supported for perturbation methods')

        input_shape = xs[0].shape
        if window_shape is not None:
            assert len(window_shape) == len(input_shape), \
                'window_shape must have length of input (%d)' % len(input_shape)
            self.window_shape = tuple(window_shape)
        else:
            self.window_shape = (1,) * len(input_shape)

        if step is not None:
            assert isinstance(step, int) or len(step) == len(input_shape), \
                'step must be integer or tuple with the length of input (%d)' % len(input_shape)
            self.step = step
        else:
            self.step = 1
        self.replace_value = 0.0
        print('Input shape: %s; window_shape %s; step %s' % (input_shape, self.window_shape, self.step))

    def run(self):
        self._run_original()

        input_shape = self.xs.shape[1:]
        batch_size = self.xs.shape[0]
        total_dim = np.asscalar(np.prod(input_shape))

        # Create mask
        index_matrix = np.arange(total_dim).reshape(input_shape)
        idx_patches = view_as_windows(index_matrix, self.window_shape, self.step).reshape((-1,) + self.window_shape)
        heatmap = np.zeros_like(self.xs, dtype=np.float32).reshape((-1), total_dim)
        w = np.zeros_like(heatmap)

        # Compute original output
        eval0 = self._run_original()

        # Start perturbation loop
        for i, p in enumerate(idx_patches):
            mask = np.ones(input_shape).flatten()
            mask[p.flatten()] = self.replace_value
            masked_xs = mask.reshape((1,) + input_shape) * self.xs
            delta = eval0 - self._run_input(masked_xs)
            delta_aggregated = np.sum(delta.reshape((batch_size, -1)), -1, keepdims=True)
            heatmap[:, p.flatten()] += delta_aggregated
            w[:, p.flatten()] += p.size

        attribution = np.reshape(heatmap / w, self.xs.shape)
        if np.isnan(attribution).any():
            warnings.warn('Attributions generated by Occlusion method contain nans, '
                          'probably because window_shape and step do not allow to cover the all input.')
        return attribution


# -----------------------------------------------------------------------------
# END ATTRIBUTION METHODS
# -----------------------------------------------------------------------------


attribution_methods = OrderedDict({
    'zero': (DummyZero, 0),
    'saliency': (Saliency, 1),
    'grad*input': (GradientXInput, 2),
    'intgrad': (IntegratedGradients, 3),
    'elrp': (EpsilonLRP, 4),
    'deeplift': (DeepLIFTRescale, 5),
    'shapley': (DeepShapley, 6),
    'linear': (Linear, 7),
    'occlusion': (Occlusion, 8)
})



@ops.RegisterGradient("DeepExplainGrad")
def deepexplain_grad(op, grad):
    global _ENABLED_METHOD_CLASS, _GRAD_OVERRIDE_CHECKFLAG
    _GRAD_OVERRIDE_CHECKFLAG = 1
    if _ENABLED_METHOD_CLASS is not None \
            and issubclass(_ENABLED_METHOD_CLASS, GradientBasedMethod):
        return _ENABLED_METHOD_CLASS.nonlinearity_grad_override(op, grad)
    else:
        return original_grad(op, grad)

@ops.RegisterGradient("MatMulDeepExplainGrad")
def matmul_deepexplain_grad(op, grad):
    global _ENABLED_METHOD_CLASS, _MATMUL_GRAD_OVERRIDE_CHECKFLAG
    _MATMUL_GRAD_OVERRIDE_CHECKFLAG = 1
    if _ENABLED_METHOD_CLASS is not None \
            and issubclass(_ENABLED_METHOD_CLASS, GradientBasedMethod):
        return _ENABLED_METHOD_CLASS.matmul_grad_override(op, grad)
    else:
        return original_grad(op, grad)

@ops.RegisterGradient("ConvolutionDeepExplainGrad")
def convolution_deepexplain_grad(op, grad):
    global _ENABLED_METHOD_CLASS, _MATMUL_GRAD_OVERRIDE_CHECKFLAG
    _MATMUL_GRAD_OVERRIDE_CHECKFLAG = 1
    if _ENABLED_METHOD_CLASS is not None \
            and issubclass(_ENABLED_METHOD_CLASS, GradientBasedMethod):
        return _ENABLED_METHOD_CLASS.convolution_grad_override(op, grad)
    else:
        return original_grad(op, grad)


@ops.RegisterGradient("MaxPoolDeepExplainGrad")
def maxpool_deepexplain_grad(op, grad):
    global _ENABLED_METHOD_CLASS, _MATMUL_GRAD_OVERRIDE_CHECKFLAG
    _MATMUL_GRAD_OVERRIDE_CHECKFLAG = 1
    if _ENABLED_METHOD_CLASS is not None \
            and issubclass(_ENABLED_METHOD_CLASS, GradientBasedMethod):
        return _ENABLED_METHOD_CLASS.maxpool_grad_override(op, grad)
    else:
        return original_grad(op, grad)


class DeepExplain(object):

    def __init__(self, graph=None, session=tf.get_default_session()):
        global SESSION
        self.method = None
        self.batch_size = None
        self.session = session
        self.graph = session.graph if graph is None else graph
        self.graph_context = self.graph.as_default()
        self.override_context = self.graph.gradient_override_map(self.get_override_map())
        self.keras_phase_placeholder = None
        self.context_on = False
        if self.session is None:
            raise RuntimeError('DeepExplain: could not retrieve a session. Use DeepExplain(session=your_session).')
        SESSION = session

    def __enter__(self):
        # Override gradient of all ops created in context
        self.graph_context.__enter__()
        self.override_context.__enter__()
        self.context_on = True
        return self

    def __exit__(self, type, value, traceback):
        self.graph_context.__exit__(type, value, traceback)
        self.override_context.__exit__(type, value, traceback)
        self.context_on = False

    def explain(self, method, T, X, xs, **kwargs):
        if not self.context_on:
            raise RuntimeError('Explain can be called only within a DeepExplain context.')
        global _ENABLED_METHOD_CLASS, _GRAD_OVERRIDE_CHECKFLAG, _MATMUL_GRAD_OVERRIDE_CHECKFLAG

        self.method = method
        if self.method in attribution_methods:
            method_class, method_flag = attribution_methods[self.method]
        else:
            raise RuntimeError('Method must be in %s' % list(attribution_methods.keys()))
        print('DeepExplain: running "%s" explanation method (%d)' % (self.method, method_flag))
        self._check_ops()
        _GRAD_OVERRIDE_CHECKFLAG = 0
        _MATMUL_GRAD_OVERRIDE_CHECKFLAG = 0

        _ENABLED_METHOD_CLASS = method_class
        method = _ENABLED_METHOD_CLASS(T, X, xs, self.session, self.keras_phase_placeholder, **kwargs)
        result = method.run()
        if issubclass(_ENABLED_METHOD_CLASS, GradientBasedMethod) and _GRAD_OVERRIDE_CHECKFLAG == 0:
            warnings.warn('DeepExplain detected you are trying to use an attribution method that requires '
                          'gradient override but the original gradient was used instead. You might have forgot to '
                          '(re)create your graph within the DeepExlain context. Results are not reliable!')
        _ENABLED_METHOD_CLASS = None
        _GRAD_OVERRIDE_CHECKFLAG = 0
        _MATMUL_GRAD_OVERRIDE_CHECKFLAG = 0

        self.keras_phase_placeholder = None
        return result

    def enable_override(self, method):
        global _ENABLED_METHOD_CLASS, _GRAD_OVERRIDE_CHECKFLAG, _MATMUL_GRAD_OVERRIDE_CHECKFLAG
        if method in attribution_methods:
            method_class, method_flag = attribution_methods[method]
            _ENABLED_METHOD_CLASS = method_class
        else:
            raise RuntimeError('Method must be in %s' % list(attribution_methods.keys()))

    def disable_override(self):
        global _ENABLED_METHOD_CLASS, _GRAD_OVERRIDE_CHECKFLAG, _MATMUL_GRAD_OVERRIDE_CHECKFLAG
        _ENABLED_METHOD_CLASS = None

    @staticmethod
    def get_override_map():
        map = dict((a, 'DeepExplainGrad') for a in SUPPORTED_ACTIVATIONS)
        #map['MatMul'] = 'MatMulDeepExplainGrad'
        #map['Conv2D'] = 'ConvolutionDeepExplainGrad'
        map['MaxPool'] = 'MaxPoolDeepExplainGrad'
        print (map)
        return map

    def _check_ops(self):
        """
        Heuristically check if any op is in the list of unsupported activation functions.
        This does not cover all cases where explanation methods would fail, and must be improved in the future.
        Also, check if the placeholder named 'keras_learning_phase' exists in the graph. This is used by Keras
         and needs to be passed in feed_dict.
        :return:
        """
        g = tf.get_default_graph()
        for op in g.get_operations():
            if len(op.inputs) > 0 and not op.name.startswith('gradients'):
                if op.type in UNSUPPORTED_ACTIVATIONS:
                    warnings.warn('Detected unsupported activation (%s). '
                                  'This might lead to unexpected or wrong results.' % op.type)
            elif 'keras_learning_phase' in op.name:
                self.keras_phase_placeholder = op.outputs[0]





