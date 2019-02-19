import tensorflow as tf
from keras import backend as K
from keras.layers import Dense, Conv2D, Flatten
from keras.engine.base_layer import Layer
from math import pi
from scipy.special import erf, erfc
import numpy as np

exp = np.exp


def rect_mean(mus, vs):
    assert mus.shape == vs.shape
    m = mus
    v = vs
    s = v**0.5
    r1 = s*exp(-(m**2)/(2*v))/(2*pi)**0.5
    r2 = 0.5*m*(1+erf(m/((2*v)**0.5)))
    return r1 + r2


def rect_var(mus, vs):
    assert mus.shape == vs.shape
    m = mus
    v = vs
    s = v**0.5
    r1 = -v*exp(-(m**2)/v)/(2*pi)
    r2 = (-m*s*erf(m/(2*v)**0.5)*exp(-(m**2)/(2*v)))/(2*pi)**0.5
    r3 = -0.25*(-2+erfc(m/(2*v)**0.5))*(2*v + m**2 *erfc(m/(2*v)**0.5))
    return r1+r2+r3



def scale_gaussians(mus, vs, factors):
    """
    Scale several gaussian distributions at the same time
    mus and vars can have arbitrary shape but it must be mus.shape == vars.shape
    and mus.shape[0] = factors.shape[0]
    :param mus: nd-array of expectations
    :param vars: nd-array array of corresponding variances
    :param factors: array of scaling factors (1D)
    :return:
    """
    assert mus.shape == vs.shape
    assert factors.shape[0] == mus.shape[0]
    assert len(factors.shape) == 1
    factors = np.expand_dims(factors, 1)
    return mus*factors, vs*(factors**2.0)


class ProbFlatten(Flatten):
    """
    Propagate distributions over a Dense layer
    """
    def __init__(self, **kwargs):
        self.n_batch = 0
        self.n_feat = 0
        self.kn = 0
        self.input_features = 0
        super(ProbFlatten, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        self.n_batch = input_shape[0]
        self.kn = input_shape[1]
        self.input_features = input_shape[2]
        self.n_feat = np.prod(input_shape[3:-1])
        return self.n_batch, self.kn, self.input_features, self.n_feat, 4

    def assert_input_compatibility(self, inputs):
        return

    def call(self, inputs):
        # Get the two input Tensors
        return K.reshape(inputs, (self.n_batch, self.kn, self.input_features, self.n_feat, 4))


class ProbConv2D(Conv2D):
    """
    Propagate distributions over a Dense layer
    """
    def __init__(self, filters, kernel_size, first=False, kn=None, **kwargs):
        self.first = first
        self.kn = kn
        self.ks = None
        super(ProbConv2D, self).__init__(filters, kernel_size, **kwargs)

    # Normal input of dense layer is [batch, feat].
    # Instead, ProbDense takes the following input:
    # - if first==True: [batch, feat, 4]
    # - if first==False: [batch, input_feat, feat, 4]
    def build(self, input_shape):
        if self.first:
            if self.kn is None:
                self.kn = input_shape[1]*input_shape[2]
            self.ks = range(0, input_shape[1]*input_shape[2], max(1, input_shape[1]*input_shape[2] // self.kn))
            self.kn = len(self.ks)
        if self.first is True:
            return super(ProbConv2D, self).build(input_shape)
        else:
            return super(ProbConv2D, self).build((input_shape[0],) + input_shape[3:])

    def compute_output_shape(self, input_shape):
        n_batch = input_shape[0]
        kn = self.kn if self.first else input_shape[1]
        n_input_feat = input_shape[1]*input_shape[2] if self.first else input_shape[2]
        conv_input_shape = input_shape
        if not self.first:
            conv_input_shape = (input_shape[0],) + input_shape[3:]
        out_shape = super(ProbConv2D, self).compute_output_shape(conv_input_shape)
        return (n_batch, kn, n_input_feat) + out_shape[1:] + (4,)

    def assert_input_compatibility(self, inputs):
        assert len(inputs.shape) == 4 if self.first else 6
        return

    def call(self, inputs):
        # Get the two input Tensors
        return tf.py_func(self.py_call, [inputs, self.kernel, self.bias], tf.float32)

    def py_call(self, inputs, kernel, bias):
        n_batch = inputs.shape[0]
        kn = self.kn if self.first else inputs.shape[1]
        n_input_feat = np.prod(inputs.shape[1:]) if self.first else inputs.shape[2]
        print (inputs.shape)
        print (self.compute_output_shape(inputs.shape))
        output = np.zeros(self.compute_output_shape(inputs.shape))

        for ki in range(kn):
            for i in range(n_input_feat):
                sample = inputs[:, ki, i] if not self.first else inputs
                # sample shape: [batch, ..., 4] if first=False, else [batch, ...]
                if self.first is True:
                    # First (input) layer
                    # Remove feature i from sample
                    sample_shape = sample.shape
                    sample_i = sample.reshape((n_batch, -1))
                    sample_i[:, i] = 0
                    sample_i = sample_i.reshape(sample_shape)
                    # Dense is essentially a dot product.
                    # Here, instead of 1, we do 3 dot products
                    dot = K.conv2d(sample, kernel, self.strides, self.padding, self.data_format, self.dilation_rate)
                    dot_i = K.conv2d(sample_i, kernel, self.strides, self.padding, self.data_format, self.dilation_rate)
                    dot_v = K.conv2d(sample_i**2, kernel**2, self.strides, self.padding, self.data_format, self.dilation_rate)
                    # Compute mean without feature i
                    Kf = np.prod(kernel.shape[0:3])
                    mu = dot_i / Kf
                    # Compensate for number of players in current coalition
                    mu1 = mu * self.ks[ki] / Kf
                    # Compute mean of the distribution that also includes player i (acting as bias to expectation)
                    mu2 = mu1 + (dot - dot_i)
                    # Compute variance without player i
                    v1 = dot_v / Kf - mu**2
                    # Compensate for number or players in the coalition
                    v1 = v1 * self.ks[ki] / Kf
                    # Set something different than 0 if necessary
                    v1 = np.maximum(0.000001, v1)
                    # Since player i is only a bias, at this point the variance of the distribution than
                    # includes it is the same
                    v2 = v1
                else:
                    assert self.first is not True
                    assert len(sample.shape) == 3
                    mu1 = sample[..., 0]
                    mu2 = sample[..., 1]
                    v1 = sample[..., 2]
                    v2 = sample[..., 3]
                    mu1 = K.conv2d(mu1, kernel, self.strides, self.padding, self.data_format, self.dilation_rate)
                    mu2 = K.conv2d(mu2, kernel, self.strides, self.padding, self.data_format, self.dilation_rate)
                    v1 = K.conv2d(v1**2, kernel**2, self.strides, self.padding, self.data_format, self.dilation_rate)
                    v2 = K.conv2d(v2**2, kernel**2, self.strides, self.padding, self.data_format, self.dilation_rate)


                # assert mu1.shape == (n_batch, n_output_feat,), mu1.shape
                # assert mu2.shape == (n_batch, n_output_feat,), mu2.shape
                # assert v1.shape == (n_batch, n_output_feat,), v1.shape
                # assert v2.shape == (n_batch, n_output_feat,), v2.shape

                # Translate if there is a bias
                if self.use_bias:
                    mu1 += bias
                    mu2 += bias

                # Apply relu if there
                if self.activation.__name__ is 'relu':
                    t = rect_mean(mu1, v1)
                    v1 = rect_var(mu1, v1)
                    mu1 = t
                    t = rect_mean(mu2, v2)
                    v2 = rect_var(mu2, v2)
                    mu2 = t
                elif self.activation.__name__ is not 'linear':
                    raise Exception("Activation can only be ReLU or linear")

                tmp = np.dstack([mu1, mu2, v1, v2])
                #assert tmp.shape == (n_batch, n_output_feat, 4), tmp.shape
                output[:, ki, i, ...] = tmp
        assert len(output.shape) == 5
        return output.astype(np.float32)


class ProbDense(Dense):
    """
    Propagate distributions over a Dense layer
    """
    def __init__(self, units, first=False, kn=None, **kwargs):
        self.first = first
        self.kn = kn
        self.ks = None
        super(ProbDense, self).__init__(units, **kwargs)

    # Normal input of dense layer is [batch, feat].
    # Instead, ProbDense takes the following input:
    # - if first==True: [batch, feat, 4]
    # - if first==False: [batch, input_feat, feat, 4]
    def build(self, input_shape):
        if self.first:
            if self.kn is None:
                self.kn = input_shape[-1]
            self.ks = range(0, input_shape[-1], max(1, input_shape[-1] // self.kn))
            self.kn = len(self.ks)
        if self.first is True:
            return super(ProbDense, self).build(input_shape)
        else:
            return super(ProbDense, self).build((input_shape[0], input_shape[3]))

    def compute_output_shape(self, input_shape):
        n_batch = input_shape[0]
        kn = self.kn if self.first else input_shape[1]
        n_input_feat = input_shape[1] if self.first else input_shape[2]
        n_output_feat = self.kernel.shape[1].value
        return n_batch, kn, n_input_feat, n_output_feat, 4

    def assert_input_compatibility(self, inputs):
        assert len(inputs.shape) == 2 if self.first else 5
        return

    def call(self, inputs):
        # Get the two input Tensors
        return tf.py_func(self.py_call, [inputs, self.kernel, self.bias], tf.float32)

    def py_call(self, inputs, kernel, bias):
        n_batch = inputs.shape[0]
        kn = self.kn if self.first else inputs.shape[1]
        n_input_feat = inputs.shape[1] if self.first else inputs.shape[2]
        n_output_feat = kernel.shape[-1]
        output = np.zeros((n_batch, kn, n_input_feat, n_output_feat, 4))

        for ki in range(kn):
            for i in range(n_input_feat):
                sample = inputs[:, ki, i] if not self.first else inputs
                # sample shape: [batch, n_feat, 4] if first=False, else [batch, n_feat]
                if len(sample.shape) == 2:
                    # First (input) layer
                    assert self.first is True
                    # Remove feature i from sample
                    sample_i = sample.copy()
                    sample_i[:, i] = 0
                    # Dense is essentially a dot product.
                    # Here, instead of 1, we do 3 dot products
                    dot = np.dot(sample, kernel)
                    dot_i = np.dot(sample_i, kernel)
                    dot_v = np.dot(sample_i ** 2, kernel ** 2)
                    # Compute mean without feature i
                    mu = dot_i / (n_input_feat - 1)
                    # Compensate for number of players in current coalition
                    mu1 = mu * self.ks[ki]
                    # Compute mean of the distribution that also includes player i (acting as bias to expectation)
                    mu2 = mu1 + (dot - dot_i)
                    # Compute variance without player i
                    v1 = dot_v / (n_input_feat - 1) - mu**2
                    # Compensate for number or players in the coalition
                    v1 = v1 * self.ks[ki]
                    # Set something different than 0 if necessary
                    v1 = np.maximum(0.000001, v1)
                    # Since player i is only a bias, at this point the variance of the distribution than
                    # includes it is the same
                    v2 = v1
                else:
                    assert self.first is not True
                    assert len(sample.shape) == 3
                    mu1 = sample[..., 0]
                    mu2 = sample[..., 1]
                    v1 = sample[..., 2]
                    v2 = sample[..., 3]
                    mu1 = np.dot(mu1, kernel)
                    mu2 = np.dot(mu2, kernel)
                    v1 = np.dot(v1, kernel ** 2)
                    v2 = np.dot(v2, kernel ** 2)

                assert mu1.shape == (n_batch, n_output_feat,), mu1.shape
                assert mu2.shape == (n_batch, n_output_feat,), mu2.shape
                assert v1.shape == (n_batch, n_output_feat,), v1.shape
                assert v2.shape == (n_batch, n_output_feat,), v2.shape

                # Translate if there is a bias
                if self.use_bias:
                    mu1 += bias
                    mu2 += bias

                # Apply relu if there
                if self.activation.__name__ is 'relu':
                    t = rect_mean(mu1, v1)
                    v1 = rect_var(mu1, v1)
                    mu1 = t
                    t = rect_mean(mu2, v2)
                    v2 = rect_var(mu2, v2)
                    mu2 = t
                elif self.activation.__name__ is not 'linear':
                    raise Exception("Activation can only be ReLU or linear")

                tmp = np.dstack([mu1, mu2, v1, v2])
                assert tmp.shape == (n_batch, n_output_feat, 4), tmp.shape
                output[:, ki, i, :] = tmp
        assert len(output.shape) == 5
        return output.astype(np.float32)


# def to_prob_distributions(dataset, ks=None):
#     """
#
#     :param dataset: [samples, features] ndarray
#     :ks optional number of coalitions (default ks = #features)
#     :return: [samples*ks, features, 4]
#     """
#     assert len(dataset.shape) == 2, "Dataset must have 2 dimensions but %s found" % str(dataset.shape)
#     n_batch, n_feat = dataset.shape
#
#     if ks is None:
#         ks = n_feat
#
#     assert 0 < ks <= n_feat, "ks must be greater than 0 and less or equal the number of features"
#
#     # Define coalition sizes
#     Xs = range(0, n_feat, max(1, n_feat // ks))
#     Xs = [1]
#
#     # Compute global statistics for each sampple
#     mu = np.mean(dataset, 1, keepdims=True)
#     var = np.var(dataset, 1, keepdims=True)
#
#
#     mu = (np.tile(np.sum(dataset, 1, keepdims=True), [1, n_feat]) - dataset) / (n_feat-1)
#     var = (np.tile(np.sum(dataset**2, 1, keepdims=True), [1, n_feat]) - dataset**2) / (n_feat-1)
#     var -= mu**2
#
#     # mu1: mean of coalitions for each sample/k-value (all features the same for now)
#     #mu1 = np.repeat(mu, n_feat, axis=-1)
#     #mu1 = mu
#     mu1 = np.tile(mu, (len(Xs), 1)) * np.expand_dims(np.repeat(Xs, n_batch), 1)
#     assert mu1.shape == (n_batch*len(Xs), n_feat)
#
#     # mu2: mean of coalitions for each sample/k-value + current feature
#     mu2 = mu1 + np.tile(dataset, (len(Xs), 1))
#     assert mu2.shape == (n_batch * len(Xs), n_feat)
#
#     # v1: variance of coalitions for each sample/k-value (all features the same for now)
#     #v1 = np.repeat(var, n_feat, axis=-1)
#     v1 = np.tile(var, (len(Xs), 1)) #/ np.expand_dims(np.repeat(Xs, n_batch), -1)
#     assert v1.shape == (n_batch * len(Xs), n_feat)
#
#     # v2: variance of coalitions for each sample/k-value  + current feature (as current feature is a bias, v1=v2_
#     v2 = np.copy(v1)
#
#     result = np.dstack([mu1, mu2, v1, v2])
#     assert result.shape == (n_batch*len(Xs), n_feat, 4), "Result shape does not match expected: %s" % str(result.shape)
#     return result

def _test():
    # print ("Test 'to_prob_distributions'")
    # x = np.array([[1, 2, 3], [-1, 0, 1]])
    # y = to_prob_distributions(x)
    #
    # assert y.shape == (2*3, 3, 4), "Expect shape (6, 3, 4) but got %s" % str(y.shape)
    # assert np.all(y[:, :, 2] == y[:, :, 3]), "Expected variances columns to be the same"
    # assert np.all(y[:2, :, 0] == 0), "Expect mu1 with k=0 to be zero"
    # assert np.all(y[:2, :, 1] == x), "Expect mu2 with k=0 to be equal to features"
    # assert np.all(y[2:4, :, 0] == np.array([[2.5, 2.0, 1.5], [0.5, 0.0, -0.5]])), "Expect mu1 with k=1 to be equal to mean of dataset --" + str(y[2:4, :, 0])
    # assert np.all(y[2:4, :, 1] == np.array([[2.5, 2.0, 1.5], [0.5, 0.0, -0.5]]) + x), "Expect mu2 with k=1 to be equal to mean of dataset"
    # assert np.all(2*y[2:4, :, 0] == y[4:6, :, 0]), "Expect mu1 with k=2 to be twice as mu1 with k=1 -- " + str(y[4:6, :, 0])
    # assert np.all(y[:, :, 1] == y[:, :, 0] + np.tile(x, (3, 1))), "Expect mu2 to always be mu1+dataset"
    # print ("--> Ok")
    #
    # print ("Test 'rect_mean'")
    # x = np.array(
    #     [
    #         [0, 1],
    #         [0, 4],
    #         [1, 4],
    #         [-1, 4],
    #         [2, 4]
    #     ]
    # )
    # y = rect_mean(x[:, 0], x[:, 1])
    # assert np.all(np.isclose(y, np.array([0.3989, 0.7978, 1.3956, 0.3956, 2.1666]),  atol=0.001)), y
    # print("--> Ok")

    print("Test 'rect_var'")
    x = np.array(
        [
            [0, 1],
            [0, 4],
            [1, 4],
            [-1, 4],
            [2, 4]
        ]
    )
    y = rect_var(x[:, 0], x[:, 1])
    assert np.all(np.isclose(y, np.array([0.3408, 1.3633, 2.2137, 0.6820, 3.0043]), atol=0.001)), y
    print("--> Ok")



if __name__== "__main__":
    _test()