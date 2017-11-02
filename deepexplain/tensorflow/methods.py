from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops, gen_math_ops, math_ops
from collections import OrderedDict


def np_activation(name):
    f = None
    if 'Relu' in name:
        f = lambda x: x * (x > 0)
    elif 'Sigmoid' in name:
        f = lambda x: 1 / (1 + np.exp(-x))
    elif 'Tanh' in name:
        f = np.tanh
    elif 'Softplus' in name:
        f = lambda x: np.log(1 + np.exp(x))
    return f


def activation(name):
    f = None
    if 'Relu' in name:
        f = tf.nn.relu
    elif 'Sigmoid' in name:
        f = tf.nn.sigmoid
    elif 'Tanh' in name:
        f = tf.nn.tanh
    elif 'Softplus' in name:
        f = tf.nn.softplus
    return f


def grad_activation(name):
    f = None
    if 'Relu' in name:
        f = lambda x: tf.where(x > 0, tf.ones_like(x), tf.zeros_like(x))
    elif 'Sigmoid' in name:
        f = lambda x: tf.exp(x) / tf.square(tf.exp(x) + 1.0)
    elif 'Tanh' in name:
        f = lambda x: 1.0 / tf.square(tf.cosh(x))
    elif 'Softplus' in name:
        f = lambda x:  tf.exp(x) / (tf.exp(x) + 1.0)
    return f



class GradientBasedMethod(object):
    def __init__(self, T, X, xs, session):
        self.T = T
        self.X = X
        self.xs = xs
        self.session = session

    def get_symbolic_attribution(self):
        return tf.gradients(self.T, self.X)[0]

    def run(self):
        attributions = self.get_symbolic_attribution()
        return self.session.run(attributions, {self.X: self.xs})

    @staticmethod
    def nonlinearity_grad_override(op, grad):
        return grad * grad_activation(op.name)(op.inputs[0])


class MethodDummyZero(GradientBasedMethod):

    def get_symbolic_attribution(self,):
        print('Dummy one')
        return tf.gradients(self.T, self.X)[0]

    @staticmethod
    def nonlinearity_grad_override(op, grad):
        input = op.inputs[0]
        return tf.zeros_like(input)


class MethodDummyOne(GradientBasedMethod):

    def get_symbolic_attribution(self,):
        return tf.gradients(self.T, self.X)[0]

    @staticmethod
    def nonlinearity_grad_override(op, grad):
        input = op.inputs[0]
        return tf.ones_like(input)


class MethodGradientXInput(GradientBasedMethod):

    def get_symbolic_attribution(self,):
        return tf.gradients(self.T, self.X)[0] * self.X




attribution_methods = {
    'zero': (MethodDummyZero, 0),
    'one': (MethodDummyOne, 1),
    'grad*input': (MethodGradientXInput, 2)
}

@ops.RegisterGradient("DeepExplainGrad")
def deepexplain_grad(op, grad):
    #print (op.get_attr("_gradient_op_type"))
    mode = tf.get_default_graph().get_tensor_by_name("deepexplain_mode:0")
    #print (mode)
    cases = OrderedDict(
        (
            (tf.equal(mode, flag), lambda: method_class.nonlinearity_grad_override(op, grad))
            for method, (method_class, flag) in attribution_methods.items()
        )
    )
    return tf.case(cases, default=lambda: grad * grad_activation(op.name)(op.inputs[0]))





class DeepExplain(object):

    def __init__(self, graph=tf.get_default_graph(), sess=tf.get_default_session()):
        self.method = None
        self.mode_flag = None
        self.batch_size = None
        self.graph = graph
        self.session = sess
        self.graph_context = self.graph.as_default()
        self.override_context = self.graph.gradient_override_map(self.get_override_map())

    def get_override_map(self):
        return {'Relu': 'DeepExplainGrad'}

    def __enter__(self):
        #Override gradient of all ops created here
        self.graph_context.__enter__()
        self.override_context.__enter__()
        self.mode_flag = tf.Variable(initial_value=-1, dtype=tf.int8, name="deepexplain_mode")
        print(self.mode_flag.name)
        print ('Override')
        return self

    def __exit__(self, type, value, traceback):
        self.graph_context.__exit__(type, value, traceback)
        self.override_context.__exit__(type, value, traceback)

    def explain(self, method, T, X, xs):
        self.method = method
        print('DeepExplain: running "%s" explanation method' % self.method)
        if self.method in attribution_methods:
            method_class, method_flag = attribution_methods[self.method]
        else:
            raise RuntimeError('Supported methods: zero, one, occlusion')
        method = method_class(T, X, xs, self.session)
        self.session.run(self.mode_flag.assign(method_flag))
        result = method.run()
        self.session.run(self.mode_flag.assign(-1))
        return result


# class BaseAttributionMethod(object):
#     def __init__(self, batch_size=1, input_batch_size=None):
#         self.model = None
#         self.batch_size = batch_size
#         self.input_batch_size = batch_size
#         self.name = 'BaseClass'
#         self.eval_x = None
#         self.eval_y = None
#         self.target_input_idx = None
#         self.target_output_idx = None
#
#     """
#     Bind model. If target input and output are not first and last layers
#     of the network, target layer indeces should also be provided.
#     """
#     def bind_model(self, model):
#         self.model = model
#         print ('Target output: %s' % self.model.layers[-1].output)
#
#     def gradient_override_map(self):
#         return {}
#
#     def sanity_check(self, eval_x, eval_y, maps):
#         pass
#
#     def target_input(self):
#         return self.model.layers[0].input
#
#     def target_output(self):
#         return self.model.layers[-1].output
#
#     def get_numeric_sensitivity(self, eval_x, eval_y, shape=None, **kwargs):
#         """
#         Return sensitivity map as numpy array
#         :param eval_x: numpy input to the model
#         :param eval_y: numpy labels
#         :param shape: if provided, the sensitivity will be reshaped to this shape
#         :return: numpy array with shape of eval_x
#         """
#         self.eval_x = eval_x
#         self.eval_y = eval_y
#         sensitivity = None
#         shape = K.int_shape(self.target_input())[1:]
#
#         #y_ = Input(batch_shape=(self.batch_size,) + tuple(eval_y.shape[1:]))  # placeholder for labels
#         print ('eval_y shape: ', eval_y.shape)
#         y_ = K.placeholder(shape=(None, ) + tuple(eval_y.shape[1:]), name='y_label')  # placeholder for labels
#         symbolic_sensitivity = self.get_symbolic_sensitivity(y_)
#         evaluate = K.function([self.model.inputs[0], y_], [symbolic_sensitivity])
#
#         for i in range(int(len(eval_x) / self.batch_size) + 1):
#             x = eval_x[self.batch_size*i:self.batch_size*(i+1)]
#             y = eval_y[self.batch_size*i:self.batch_size*(i+1)]
#             self.input_batch_size = len(x)
#             if self.input_batch_size > 0:
#                 tmp = evaluate([x, y])[0]
#                 if sensitivity is None:
#                     sensitivity = tmp
#                 else:
#                     sensitivity = np.append(sensitivity, tmp, axis=0)
#
#         if shape is not None:
#             sensitivity = sensitivity.reshape((self.eval_x.shape[0],) + shape)
#
#         # Run sanity check if available
#         self.sanity_check(eval_x, eval_y, sensitivity)
#         return sensitivity
#
#     def get_symbolic_sensitivity(self, eval_y):
#         """
#         Return sensitivity map as numpy array
#         :param eval_y: placeholder (Tensor) for labels
#         :return: Tensor with shape of eval_x
#         """
#         raise RuntimeError('Calling get_symbolic_sensitivity on abstract class')



