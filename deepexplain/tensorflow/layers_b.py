import tensorflow as tf
from tensorflow.contrib import distributions as dist
from keras import backend as K
from keras.layers import Dense, Conv2D, Flatten, Activation, MaxPooling2D
from keras.engine.base_layer import Layer
from math import pi
from scipy.stats import norm
import numpy as np

exp = np.exp
normal = dist.Normal(loc=0., scale=1.)

# def rect_mean(mus, vs):
#     assert mus.shape == vs.shape
#     m = mus
#     v = vs
#     s = v**0.5
#     return mus*norm.cdf(mus/s) + s*norm.pdf(mus/s)
#
# def rect_var(mus, vs, mus_relu):
#     assert mus.shape == vs.shape
#     m = mus
#     v = vs
#     s = v**0.5
#     r1 = (mus**2 + v)*norm.cdf(mus/s)
#     r2 = (mus*s)*norm.pdf(mus/s)
#     return r1+r2-mus_relu**2


def tf_rect_mean(m, v):
    s = v**0.5
    return m*normal.cdf(m/s) + s*normal.prob(m/s)

def tf_rect_var(m, v, m_relu):
    s = v**0.5
    r1 = (m**2 + v)*normal.cdf(m/s)
    r2 = (m*s)*normal.prob(m/s)
    return r1+r2-m_relu**2


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
        super(ProbFlatten, self).__init__(**kwargs)
        self.n_batch = None
        self.n_feat = None

    def compute_output_shape(self, input_shape):
        self.n_batch = input_shape[0]
        self.n_feat = np.prod(input_shape[1:-1])
        return self.n_batch, self.n_feat, 2

    def assert_input_compatibility(self, inputs):
        return super(ProbFlatten, self).assert_input_compatibility(inputs[..., 0])

    def call(self, inputs):
        n_batch = tf.shape(inputs)[0]
        return K.reshape(inputs, (n_batch, -1, 2))



class ProbDense(Dense):
    """
    Propagate distributions over a Dense layer
    """
    def __init__(self, units, **kwargs):
        super(ProbDense, self).__init__(units, **kwargs)

    # Normal input of dense layer is [batch, feat].
    # Instead, ProbDense takes the following input: [batch, feat, 2] to account for mean and variance
    def build(self, input_shape):
        return super(ProbDense, self).build(input_shape[:-1])

    def compute_output_shape(self, input_shape):
        original_output_shape = super(ProbDense, self).compute_output_shape(input_shape[:-1])
        return original_output_shape + (2,)

    def assert_input_compatibility(self, inputs):
        assert len(inputs.shape) == 3, 'Input must have shape of len 3'
        return

    def call(self, inputs):
        mus = inputs[..., 0]
        vs = inputs[..., 1]
        _mu = K.dot(mus, self.kernel)
        _v = K.dot(vs, self.kernel ** 2)

        # Translate if there is a bias
        if self.use_bias:
            _mu += self.bias

        # Apply relu if there
        if self.activation.__name__ is 'relu':
            t = tf_rect_mean(_mu, _v)
            _v = tf_rect_var(_mu, _v, t)
            _mu = t
        elif self.activation.__name__ is not 'linear':
            raise Exception("Activation can only be ReLU or linear")
        return tf.stack([_mu, _v], -1)


class ProbConv2DInput():
    def __init__(self, keras_conv_layer, input_shape):
        self.l = keras_conv_layer
        self.input_shape = input_shape
        self.n_feat = np.prod(input_shape[1:]).astype('float32')
        self.sess = K.get_session()
        self.k = K.placeholder((), dtype='float32')
        self.result = None
        self.tf_inputs = K.placeholder(self.input_shape, dtype='float32')
        self.tf_inputs_i = K.placeholder(self.input_shape, dtype='float32')
        self.tf_ghost_i = K.placeholder(self.input_shape, dtype='float32')
        self.init()

    def __call__(self, inputs, k, feat_idx):
        #print ('call')
        inputs_i = inputs.copy().reshape((self.input_shape[0], -1))
        ghost_i = np.ones_like(inputs_i)
        inputs_i[:, feat_idx] = 0.0
        ghost_i[:, feat_idx] = 0.0
        inputs_i = inputs_i.reshape(self.input_shape)
        ghost_i = ghost_i.reshape(self.input_shape)
        #print ('about to run')
        return self.sess.run(self.result, feed_dict={
            self.tf_inputs: inputs,
            self.tf_inputs_i: inputs_i,
            self.tf_ghost_i: ghost_i,
            self.k: float(k)
        })

    def init(self):
        dot = K.conv2d(self.tf_inputs, self.l.kernel, self.l.strides, self.l.padding, self.l.data_format, self.l.dilation_rate)
        dot_i = K.conv2d(self.tf_inputs_i, self.l.kernel, self.l.strides, self.l.padding, self.l.data_format, self.l.dilation_rate)
        dot_v = K.conv2d(self.tf_inputs_i ** 2, self.l.kernel ** 2, self.l.strides, self.l.padding, self.l.data_format, self.l.dilation_rate)
        dot_mask = K.conv2d(self.tf_ghost_i, tf.ones_like(self.l.kernel), self.l.strides, self.l.padding, self.l.data_format, self.l.dilation_rate)
        # Compute mean without feature i
        mu = dot_i / dot_mask
        # Compensate for number of players in current coalition
        mu1 = mu * dot_mask * (self.k / self.n_feat)
        # Compute mean of the distribution that also includes player i (acting as bias to expectation)
        mu2 = mu1 + (dot - dot_i)
        # Compute variance without player i
        v1 = dot_v / dot_mask - mu ** 2
        # Compensate for number or players in the coalition
        v1 = v1 * dot_mask * (self.k / self.n_feat)
        # Set something different than 0 if necessary
        v1 = K.maximum(0.00001, v1)
        # Since player i is only a bias, at this point the variance of the distribution than
        # includes it is the same
        v2 = v1

        if self.l.use_bias:
            mu1 = K.bias_add(
                mu1,
                self.l.bias,
                data_format=self.l.data_format)
            mu2 = K.bias_add(
                mu2,
                self.l.bias,
                data_format=self.l.data_format)
        print ('set result')

        self.result = tf.stack([mu1, mu2, v1, v2], -1)




class ProbConv2D(Conv2D):
    """
    Propagate distributions over a Conv2D layer
    """
    def __init__(self, filters, kernel_size, **kwargs):
        super(ProbConv2D, self).__init__(filters, kernel_size, **kwargs)

    def build(self, input_shape):
        return super(ProbConv2D, self).build(input_shape[:-1])

    def compute_output_shape(self, input_shape):
        original_output_shape = super(ProbConv2D, self).compute_output_shape(input_shape[:-1])
        return original_output_shape + (2,)

    def assert_input_compatibility(self, inputs):
        return super(ProbConv2D, self).assert_input_compatibility(inputs[..., 0])

    def call(self, inputs):
        mus = inputs[..., 0]
        vs = inputs[..., 1]

        outputs_mu = K.conv2d(
            mus,
            self.kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)

        outputs_vs = K.conv2d(
            vs,
            self.kernel**2.0,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)

        if self.use_bias:
            outputs_mu = K.bias_add(
                outputs_mu,
                self.bias,
                data_format=self.data_format)

        if self.activation.__name__ is 'relu':
            t = tf_rect_mean(outputs_mu, outputs_vs)
            outputs_vs = tf_rect_var(outputs_mu, outputs_vs, t)
            outputs_mu = t
        elif self.activation.__name__ is not 'linear':
            raise Exception("Activation can only be ReLU or linear")

        return tf.stack([outputs_mu, outputs_vs], -1)


class ProbActivationRelu(Activation):
    def __init__(self, **kwargs):
        super(ProbActivationRelu, self).__init__('relu', **kwargs)
        self.supports_masking = True

    def call(self, inputs):
        _mu = inputs[..., 0]
        _v = inputs[..., 1]
        t = tf_rect_mean(_mu, _v)
        _v = tf_rect_var(_mu, _v, t)
        _mu = t
        return tf.stack([_mu, _v], -1)


class ProbMaxPooling2D(MaxPooling2D):

    def assert_input_compatibility(self, inputs):
        return super(ProbMaxPooling2D, self).assert_input_compatibility(inputs[..., 0])

    def compute_output_shape(self, input_shape):
        original_output_shape = super(ProbMaxPooling2D, self).compute_output_shape(input_shape[:-1])
        return original_output_shape + (2,)

    def extract_patches(self, x):
        return tf.extract_image_patches(
            x,
            ksizes=(1,) + self.pool_size + (1,),
            strides=(1,) + self.strides + (1,),
            padding='VALID',
            rates=[1, 1, 1, 1]
        )

    def extract_patches_inverse(self, x, y):
        _x = tf.zeros_like(x)
        # print (_x)
        _y = self.extract_patches(_x)
        # print (_y)
        y = tf.check_numerics(
            y,
            'y contains nans',
        )
        grad = tf.gradients(_y, _x)[0]
        # Divide by grad, to "average" together the overlapping patches
        # otherwise they would simply sum up
        return tf.gradients(_y, _x, grad_ys=y)[0] / grad


    def _ab_max_pooling(self, a, b):
        mu_a = a[..., 0]
        va = a[..., 1]
        mu_b = b[..., 0]
        vb = b[..., 1]
        vavb = tf.maximum(((va + vb) ** 0.5), 0.0001)
        muamub = mu_a - mu_b
        muamub_p = mu_a + mu_b
        alpha = muamub / vavb

        mu_c =  vavb * normal.prob(alpha) + muamub*normal.cdf(alpha) + mu_b
        vc = muamub_p * vavb * normal.prob(alpha)
        vc += (mu_a ** 2 + va) * normal.cdf(alpha) + (mu_b ** 2 + vb) * (1. - normal.cdf(alpha)) - mu_c ** 2
        return tf.stack([mu_c, vc], -1)


    def _pooling_function(self, inputs, pool_size, strides,
                          padding, data_format):
        _mu = inputs[..., 0]
        _v = inputs[..., 1]

        outputs_mu = self.extract_patches(_mu)
        outputs_vs = self.extract_patches(_v)
        #print ("After extract patches")
        #print (outputs_mu)
        # Reshape patches to have all elements involved in convolution together in the last dimension
        shape = tf.shape(outputs_mu)
        input_channels = _mu.shape[-1]
        outputs_mu = tf.reshape(outputs_mu, (-1, np.prod(self.pool_size), input_channels))
        outputs_vs = tf.reshape(outputs_vs, (-1, np.prod(self.pool_size), input_channels))

        outputs_mu = tf.transpose(outputs_mu, (0, 2, 1))
        outputs_vs = tf.transpose(outputs_vs, (0, 2, 1))

        outputs_mu = tf.reshape(outputs_mu, (-1, np.prod(self.pool_size)))
        outputs_vs = tf.reshape(outputs_vs, (-1, np.prod(self.pool_size)))
        #print("After reshape")
        #print(outputs_mu)
        # Transpose


        outputs_mu = tf.transpose(outputs_mu)
        outputs_vs = tf.transpose(outputs_vs)
        #print("After transpose")
        #print(outputs_mu)

        # Apply max pooling in sequence
        t = tf.stack([outputs_mu, outputs_vs], -1)

        t = tf.scan(self._ab_max_pooling, t, reverse=True)
        outputs_mu = t[..., 0]
        outputs_vs = t[..., 1]
        #outputs_vs = tf.scan(self._ab_max_pooling_var, t, reverse=True)[..., 1]

        #print("After scan")
        #print(outputs_mu)

        #outputs_vs = tf.scan(self._ab_max_pooling_var, outputs_vs, reverse=True)

        # Set all elements = accumulator
        outputs_mu = tf.stack([outputs_mu[0, :]] * np.prod(self.pool_size), 0)
        outputs_vs = tf.stack([outputs_vs[0, :]] * np.prod(self.pool_size), 0)

        #print("After stack")
        #print(outputs_mu)

        # Transpose
        outputs_mu = tf.transpose(outputs_mu)
        outputs_vs = tf.transpose(outputs_vs)

        outputs_mu = tf.reshape(outputs_mu, (-1, input_channels, np.prod(self.pool_size)))
        outputs_vs = tf.reshape(outputs_vs, (-1, input_channels, np.prod(self.pool_size)))

        outputs_mu = tf.transpose(outputs_mu, (0, 2, 1))
        outputs_vs = tf.transpose(outputs_vs, (0, 2, 1))

        outputs_mu = tf.reshape(outputs_mu, shape)
        outputs_vs = tf.reshape(outputs_vs, shape)

        #print (outputs_mu)
        outputs_mu = self.extract_patches_inverse(_mu, outputs_mu)
        outputs_vs = self.extract_patches_inverse(_v, outputs_vs)
        #print("After parches inverse")
        #print (outputs_mu)

        outputs_mu = K.pool2d(outputs_mu, self.pool_size, self.strides, self.padding, self.data_format, pool_mode='avg')
        outputs_vs = K.pool2d(outputs_vs, self.pool_size, self.strides, self.padding, self.data_format, pool_mode='avg')

        #print("After avg pool")
        #print (outputs_mu)
        return tf.stack([outputs_mu, outputs_vs], -1)




    # def py_call(self, inputs, kernel, bias):
    #     n_batch = inputs.shape[0]
    #     kn = self.kn if self.first else inputs.shape[1]
    #     n_input_feat = inputs.shape[1] if self.first else inputs.shape[2]
    #     n_output_feat = kernel.shape[-1]
    #     output = np.zeros((n_batch, kn, n_input_feat, n_output_feat, 4))
    #
    #     for ki in range(kn):
    #         for i in range(n_input_feat):
    #             sample = inputs[:, ki, i] if not self.first else inputs
    #             # sample shape: [batch, n_feat, 4] if first=False, else [batch, n_feat]
    #             if len(sample.shape) == 2:
    #                 # First (input) layer
    #                 assert self.first is True
    #                 # Remove feature i from sample
    #                 sample_i = sample.copy()
    #                 sample_i[:, i] = 0
    #                 # Dense is essentially a dot product.
    #                 # Here, instead of 1, we do 3 dot products
    #                 dot = np.dot(sample, kernel)
    #                 dot_i = np.dot(sample_i, kernel)
    #                 dot_v = np.dot(sample_i ** 2, kernel ** 2)
    #                 # Compute mean without feature i
    #                 mu = dot_i / (n_input_feat - 1)
    #                 # Compensate for number of players in current coalition
    #                 mu1 = mu * self.ks[ki]
    #                 # Compute mean of the distribution that also includes player i (acting as bias to expectation)
    #                 mu2 = mu1 + (dot - dot_i)
    #                 # Compute variance without player i
    #                 v1 = dot_v / (n_input_feat - 1) - mu**2
    #                 # Compensate for number or players in the coalition
    #                 v1 = v1 * self.ks[ki]
    #                 # Set something different than 0 if necessary
    #                 v1 = np.maximum(0.000001, v1)
    #                 # Since player i is only a bias, at this point the variance of the distribution than
    #                 # includes it is the same
    #                 v2 = v1
    #             else:
    #                 assert self.first is not True
    #                 assert len(sample.shape) == 3
    #                 mu1 = sample[..., 0]
    #                 mu2 = sample[..., 1]
    #                 v1 = sample[..., 2]
    #                 v2 = sample[..., 3]
    #                 mu1 = np.dot(mu1, kernel)
    #                 mu2 = np.dot(mu2, kernel)
    #                 v1 = np.dot(v1, kernel ** 2)
    #                 v2 = np.dot(v2, kernel ** 2)
    #
    #             assert mu1.shape == (n_batch, n_output_feat,), mu1.shape
    #             assert mu2.shape == (n_batch, n_output_feat,), mu2.shape
    #             assert v1.shape == (n_batch, n_output_feat,), v1.shape
    #             assert v2.shape == (n_batch, n_output_feat,), v2.shape
    #
    #             # Translate if there is a bias
    #             if self.use_bias:
    #                 mu1 += bias
    #                 mu2 += bias
    #
    #             # Apply relu if there
    #             if self.activation.__name__ is 'relu':
    #                 t = rect_mean(mu1, v1)
    #                 v1 = rect_var(mu1, v1)
    #                 mu1 = t
    #                 t = rect_mean(mu2, v2)
    #                 v2 = rect_var(mu2, v2)
    #                 mu2 = t
    #             elif self.activation.__name__ is not 'linear':
    #                 raise Exception("Activation can only be ReLU or linear")
    #
    #             tmp = np.dstack([mu1, mu2, v1, v2])
    #             assert tmp.shape == (n_batch, n_output_feat, 4), tmp.shape
    #             output[:, ki, i, :] = tmp
    #     assert len(output.shape) == 5
    #     return output.astype(np.float32)


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
    import keras
    from keras.models import Sequential
    print (" Test ProbMaxPooling2D")

    mus = np.array([[1, 2], [3, 4]])
    vs = np.array([[1, 1], [1, 1]])

    x = np.array([np.stack([mus, vs], -1)])
    x = np.expand_dims(x, -2)
    print (x)
    print (x.shape)

    max_pool = ProbMaxPooling2D(2)

    model = Sequential()
    model.add(max_pool)
    y = model.predict(x)
    print ("Prediction")
    print (y)
    print (y.shape)

    #a = np.array([[3, 6]]).astype('float32')
    #b = np.array([[5, 0]]).astype('float32')
    #print (max_pool._ab_max_pooling(a, b).eval(session=K.get_session()))
    # print (t)




if __name__== "__main__":
    _test()