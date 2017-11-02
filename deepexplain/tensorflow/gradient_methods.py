import keras.backend as K
import os, sys
from termcolor import colored
import numpy as np
from keras.models import Model
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops, gen_math_ops, math_ops
references = {}

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

#@ops.RegisterGradient("GradLRP")
def _GradLRP(op, grad):
    output = op.outputs[0]
    input = op.inputs[0]
    return grad * output / (input + 1e-6 *
                            tf.where(input >= 0, tf.ones_like(input), -1 * tf.ones_like(input)))


#@ops.RegisterGradient("GradGUIDED_BP")
def _GradGUIDED_BP(op, grad):
    input = op.inputs[0]
    return grad * tf.where(0. < grad, grad_activation('Relu')(input), tf.zeros_like(input))


#@ops.RegisterGradient("GradDEEP_LIFT")
def _GradDEEP_LIFT(op, grad):
    global references
    output = op.outputs[0]
    input = op.inputs[0]
    ref_input = references[op.name]
    ref_output = activation(op.name)(ref_input)
    delta_out = output - ref_output
    delta_in = input - ref_input
    return grad * tf.where(tf.abs(delta_in) > 1e-6, delta_out / delta_in, grad_activation(op.name)(0.5 * (ref_input + input)))

#@ops.RegisterGradient("BypassMaxPoolGrad")
def BypassMaxPoolGrad(op, grad):
    output = op.outputs[0]
    input = op.inputs[0]
    return gen_nn_ops._avg_pool_grad(tf.shape(input), grad,
                                     ksize=op.get_attr('ksize'),
                                     strides=op.get_attr('strides'),
                                     padding=op.get_attr('padding'),
                                     data_format=op.get_attr('data_format'))


def deepexplain_zerograd(op, grad):
    input = op.inputs[0]
    return tf.zeros_like(input)

def deepexplain_onegrad(op, grad):
    input = op.inputs[0]
    return tf.ones_like(input)

@ops.RegisterGradient("DeepExplainGrad")
def deepexplain_grad(op, grad):
    mode = tf.get_default_graph().get_tensor_by_name("deepexplain_mode:0")
    print (mode)
    return tf.case({
        tf.equal(mode, 0): lambda: deepexplain_zerograd(op, grad),
        tf.equal(mode, 1): lambda:  deepexplain_onegrad(op, grad)
    }, default=lambda: grad)



# @ops.RegisterGradient("GradFLOW_TRANSLATE")
# def _GradFlowTranslate(op, grad):
#     print ('Flow Translate Gradient')
#     output = op.outputs[0]
#     input = op.inputs[0]
#     bias = tf.zeros_like(op.inputs[0])
#     if 'BiasAdd' in op.inputs[0].op.name:
#         # BiasAdd(values, bias) performs sum(values) + bias
#         bias = op.inputs[0].op.inputs[1]
#     f = activation(op.name)
#     return grad * (output - f(tf.zeros_like(output))) / ((input - bias) + 1e-4 * tf.where(input - bias >= 0, tf.ones_like(input), -1 * tf.ones_like(input)))

#
# class BPMethod(BaseAttributionMethod):
#     def __init__(self, mode, bypass_maxpool=False, **kwargs):
#         super(BPMethod, self).__init__(**kwargs)
#         self.mode = mode
#         self.bypass_maxpool = bypass_maxpool
#         self.name = ''
#         if self.mode == 'lrp': self.name = 'LRP*'
#         elif self.mode == 'deep_lift': self.name = 'DeepLIFT*'
#         elif self.mode == 'guided_bp': self.name = 'GuidedBP'
#         else: self.name = self.mode + '*'
#         if self.bypass_maxpool:
#             self.name += '_b'
#
#     """
#     Override all activation gradients that might be used in a model,
#     each with a specialized function according to the chosen attribution method.
#     """
#     def gradient_override_map(self):
#         print ('Gradient oveeride: ' + self.mode.upper())
#         grad_name = 'Grad' + self.mode.upper()
#         override_map = {
#             'Sigmoid': grad_name,
#             'Relu': grad_name,
#             'Softplus': grad_name,
#             'Tanh': grad_name}
#         if self.bypass_maxpool:
#             override_map['MaxPool'] = 'BypassMaxPoolGrad'
#         return override_map
#
#
#     def get_symbolic_sensitivity(self, eval_y=None):
#         target_input = self.target_input()
#         output = self.target_output()
#         output = output * eval_y
#         return tf.gradients(output, target_input)[0] * target_input
#
#     def get_numeric_sensitivity(self, eval_x, eval_y, shape=None, **kwargs):
#         if self.mode == 'deep_lift':
#             self._init_references(eval_x[0].shape)
#         return super(BPMethod, self).get_numeric_sensitivity(eval_x, eval_y, shape)
#
#     def sanity_check(self, eval_x, eval_y, maps):
#         if self.mode == 'deep_lift':
#             target_output = self.target_output()
#             data_input = self.model.inputs[0]
#             f = K.function([data_input], [target_output])
#             len_test = 1
#             y = np.sum(f([eval_x[:len_test]])[0] * eval_y[:len_test])
#             y0 = np.sum(f([np.zeros_like(eval_x[:len_test])])[0] * eval_y[:len_test])
#             ex = np.sum(maps[:len_test])
#             if (np.isnan(ex)):
#                 print(colored('Deep LIFT: heatmap contains nans', 'red', attrs=['bold']))
#             if np.abs(ex - (y - y0)) > 0.1:
#                 print('Summation check target output: ', target_output)
#                 print('Sum x: %.3f' % y)
#                 print('Sum x0: %.3f' % y0)
#                 print('Sum x - x0: %.3f' % (y - y0))
#                 print('Sum maps: %.3f' % ex)
#                 print (colored('Deep LIFT: summation check failed!', 'red', attrs=['bold']))
#         elif self.mode == 'lrp':
#             ex = np.sum(maps[:1])
#             if (np.isnan(ex)):
#                 print(colored('LRP: heatmap contains nans', 'red', attrs=['bold']))
#
#     """
#     Run the network on a specific input (eg. zero input) and store all
#     inputs to the activation functions. This is necessary to implement DeepLift.
#     """
#     def _init_references(self, x_shape):
#         # Use a global dictionary to store the activations
#         print ('DeepLIFT: computing references...')
#         sys.stdout.flush()
#         global references
#         references.clear()
#         ops = []
#         g = tf.get_default_graph()
#         for op in g.get_operations():
#             #print (op, op.inputs)
#             if len(op.inputs) > 0 and not op.name.startswith('gradients'):
#                 if any(s in op.name for s in ['Relu', 'Sigmoid', 'Tanh', 'Softplus']):
#                     ops.append(op)
#         print ('DeepLIFT ref ops:')
#         YR = K.function([self.model.inputs[0]], [o.inputs[0] for o in ops])([np.zeros((1,) + x_shape)])
#         for (r, op) in zip(YR, ops):
#             references[op.name] = r
#         print('DeepLIFT: references ready')
#         sys.stdout.flush()
