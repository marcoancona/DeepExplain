from unittest import TestCase
import pkg_resources
import logging, warnings
import tensorflow as tf
from tensorflow.contrib.framework import is_tensor
import numpy as np

from deepexplain.tensorflow import DeepExplain
from deepexplain.tensorflow.methods import original_grad # test only

activations = {'Relu': tf.nn.relu,
                'Sigmoid': tf.nn.sigmoid,
                'Softplus': tf.nn.softplus,
                'Tanh': tf.nn.tanh}


def simple_model(activation, session):
    X = tf.placeholder("float", [None, 2])
    w1 = tf.Variable(initial_value=[[1.0, -1.0], [-1.0, 1.0]])
    b1 = tf.Variable(initial_value=[1.5, -1.0])
    w2 = tf.Variable(initial_value=[[1.1, 1.4], [-0.5, 1.0]])
    b2 = tf.Variable(initial_value=[0.0, 2.0])

    layer1 = activation(tf.matmul(X, w1) + b1)
    out = tf.matmul(layer1, w2) + b2
    session.run(tf.global_variables_initializer())
    return X, out


def simpler_model(session):
    """
    Implements ReLU( ReLU(x1 - 1) - ReLU(x2) )
    :
    """
    X = tf.placeholder("float", [None, 2])
    w1 = tf.Variable(initial_value=[[1.0, 0.0], [0.0, 1.0]], trainable=False)
    b1 = tf.Variable(initial_value=[-1.0, 0], trainable=False)
    w2 = tf.Variable(initial_value=[[1.0], [-1.0]], trainable=False)

    l1 = tf.nn.relu(tf.matmul(X, w1) + b1)
    out = tf.nn.relu(tf.matmul(l1, w2))
    session.run(tf.global_variables_initializer())
    return X, out


def min_model(session):
    """
    Implements min(xi)
    """
    X = tf.placeholder("float", [None, 2])
    out = tf.reduce_min(X,1)
    session.run(tf.global_variables_initializer())
    return X, out


def min_model_2d(session):
    """
    Implements min(xi)
    """
    X = tf.placeholder("float", [None, 2, 2])
    out = tf.reduce_min(tf.reshape(X, (-1, 4)), 1)
    session.run(tf.global_variables_initializer())
    return X, out


def simple_multi_inputs_model(session):
    """
    Implements Relu (3*x1|2*x2) | is a concat op
    :
    """
    X1 = tf.placeholder("float", [None, 2])
    X2 = tf.placeholder("float", [None, 2])
    w1 = tf.Variable(initial_value=[[3.0, 0.0], [0.0, 3.0]], trainable=False)
    w2 = tf.Variable(initial_value=[[2.0, 0.0], [0.0, 2.0]], trainable=False)

    out = tf.nn.relu(tf.concat([X1*w1, X2*w2], 1))
    session.run(tf.global_variables_initializer())
    return X1, X2, out


def simple_multi_inputs_model2(session):
    """
    Implements Relu (3*x1|2*x2) | is a concat op
    :
    """
    X1 = tf.placeholder("float", [None, 2])
    X2 = tf.placeholder("float", [None, 1])
    w1 = tf.Variable(initial_value=[3.0], trainable=False)
    w2 = tf.Variable(initial_value=[2.0], trainable=False)

    out = tf.nn.relu(tf.concat([X1*w1, X2*w2], 1))
    session.run(tf.global_variables_initializer())
    return X1, X2, out


def train_xor(session):
    # Since setting seed is not always working on TF, initial weights values are hardcoded for reproducibility
    X = tf.placeholder("float", [None, 2])
    Y = tf.placeholder("float", [None, 1])
    w1 = tf.Variable(initial_value=[[0.10711301, -0.0987727], [-1.57625198, 1.34942603]])
    b1 = tf.Variable(initial_value=[-0.30955192, -0.14483099])
    w2 = tf.Variable(initial_value=[[0.69259691], [-0.16255915]])
    b2 = tf.Variable(initial_value=[1.53952825])

    l1 = tf.nn.relu(tf.matmul(X, w1) + b1)
    out = tf.matmul(l1, w2) + b2
    session.run(tf.global_variables_initializer())

    # Define loss and optimizer
    loss = tf.reduce_mean(tf.losses.mean_squared_error(Y, out))
    train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

    # Generate dataset random
    np.random.seed(10)
    x = np.random.randint(0, 2, size=(10, 2))
    y = np.expand_dims(np.logical_or(x[:, 0], x[:, 1]), -1)
    l = None
    for _ in range(100):
        l, _, = session.run([loss, train_step], feed_dict={X: x, Y: y})
        #logging.critical(l)
    #logging.critical('Done')
    return np.abs(l - 0.1) < 0.01


class TestDeepExplainGeneralTF(TestCase):

    def setUp(self):
        self.session = tf.Session()

    def tearDown(self):
        self.session.close()
        tf.reset_default_graph()

    def test_tf_available(self):
        try:
            pkg_resources.require('tensorflow>=1.0.0')
        except Exception:
            try:
                pkg_resources.require('tensorflow-gpu>=1.0.0')
            except Exception:
                self.fail("Tensorflow requirement not met")

    def test_simple_model(self):
        X, out = simple_model(tf.nn.relu, self.session)
        xi = np.array([[1, 0]])
        r = self.session.run(out, {X: xi})
        self.assertEqual(r.shape, xi.shape)
        np.testing.assert_equal(r[0], [2.75,  5.5])

    def test_simpler_model(self):
        X, out = simpler_model(self.session)
        xi = np.array([[3.0, 1.0]])
        r = self.session.run(out, {X: xi})
        self.assertEqual(r.shape, (xi.shape[0], 1))
        np.testing.assert_equal(r[0], [1.0])

    def test_training(self):
        session = tf.Session()
        r = train_xor(session)
        self.assertTrue(r)

    def test_context(self):
        """
        DeepExplain overrides nonlinearity gradient
        """
        # No override
        from deepexplain.tensorflow import DeepExplain

        X = tf.placeholder("float", [None, 1])
        for name in activations:
            x1 = activations[name](X)
            x1_g = tf.gradients(x1, X)[0]
            self.assertEqual(x1_g.op.type, '%sGrad' % name)

        # Override (note: that need to pass graph! Multiple thread testing??)
        with DeepExplain(graph=tf.get_default_graph(), session=self.session) as de:
            for name in activations:
                # Gradients of nonlinear ops are overriden
                x2 = activations[name](X)
                self.assertEqual(x2.op.get_attr('_gradient_op_type').decode('utf-8'), 'DeepExplainGrad')

    def test_mismatch_input_lens(self):
        with DeepExplain(graph=tf.get_default_graph(), session=self.session) as de:
            X1 = tf.placeholder("float", [None, 1])
            X2 = tf.placeholder("float", [None, 1])
            w1 = tf.Variable(initial_value=[[0.10711301]])
            w2 = tf.Variable(initial_value=[[0.69259691]])
            out = tf.matmul(X1, w1) + tf.matmul(X2, w2)

            self.session.run(tf.global_variables_initializer())
            with self.assertRaises(RuntimeError) as cm:
                de.explain('grad*input', out, [X1, X2], [[1], [2], [3]])
            self.assertIn(
                'List of input tensors and input data have different lengths',
                str(cm.exception)
            )

    def test_multiple_inputs(self):
        with DeepExplain(graph=tf.get_default_graph(), session=self.session) as de:
            X1 = tf.placeholder("float", [None, 1])
            X2 = tf.placeholder("float", [None, 1])
            w1 = tf.Variable(initial_value=[[10.0]])
            w2 = tf.Variable(initial_value=[[10.0]])
            out = tf.matmul(X1, w1) + tf.matmul(X2, w2)

            self.session.run(tf.global_variables_initializer())
            attributions = de.explain('grad*input', out, [X1, X2], [[[2]], [[3]]])
            self.assertEqual(len(attributions), 2)
            self.assertEqual(attributions[0][0], 20.0)
            self.assertEqual(attributions[1][0], 30.0)

    def test_supported_activations(self):
        X = tf.placeholder("float", [None, 3])
        with DeepExplain(graph=tf.get_default_graph(), session=self.session) as de:
            xi = [[-1, 0, 1]]
            Y = tf.nn.relu(X)
            r = self.session.run(Y, {X: xi})
            np.testing.assert_almost_equal(r[0], [0, 0, 1], 7)
            Y = tf.nn.elu(X)
            r = self.session.run(Y, {X: xi})
            np.testing.assert_almost_equal(r[0], [-0.632120558, 0, 1], 7)
            Y = tf.nn.sigmoid(X)
            r = self.session.run(Y, {X: xi})
            np.testing.assert_almost_equal(r[0], [0.268941421, 0.5, 0.731058578], 7)
            Y = tf.nn.tanh(X)
            r = self.session.run(Y, {X: xi})
            np.testing.assert_almost_equal(r[0], [-0.761594155, 0, 0.761594155], 7)
            Y = tf.nn.softplus(X)
            r = self.session.run(Y, {X: xi})
            np.testing.assert_almost_equal(r[0], [0.313261687, 0.693147181, 1.31326168], 7)

    def test_original_grad(self):
        X = tf.placeholder("float", [None, 3])
        for name in activations:
            Y = activations[name](X)
            grad = original_grad(Y.op, tf.ones_like(X))
            self.assertTrue('Tensor' in str(type(grad)))


    def test_warning_unsupported_activations(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with DeepExplain(graph=tf.get_default_graph(), session=self.session) as de:
                X = tf.placeholder("float", [None, 3])
                Y = tf.nn.relu6(X)  # < an unsupported activation
                xi = [[-1, 0, 1]]
                de.explain('elrp', Y, X, xi)
                assert any(["unsupported activation" in str(wi.message) for wi in w])

    def test_override_as_default(self):
        """
        In DeepExplain context, nonlinearities behave as default, including training time
        """
        with DeepExplain(graph=tf.get_default_graph(), session=self.session) as de:
            r = train_xor(self.session)
            self.assertTrue(r)

    def test_explain_not_in_context(self):
        with DeepExplain(graph=tf.get_default_graph(), session=self.session) as de:
            pass
        with self.assertRaises(RuntimeError) as cm:
            de.explain('grad*input', None, None, None)
        self.assertEqual(
            'Explain can be called only within a DeepExplain context.',
            str(cm.exception)
        )

    def test_invalid_method(self):
        with DeepExplain(graph=tf.get_default_graph(), session=self.session) as de:
            with self.assertRaises(RuntimeError) as cm:
                de.explain('invalid', None, None, None)
            self.assertIn('Method must be in',
                str(cm.exception)
            )

    def test_gradient_was_not_overridden(self):
        X = tf.placeholder("float", [None, 3])
        Y = tf.nn.relu(X)
        with DeepExplain(graph=tf.get_default_graph(), session=self.session) as de:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                de.explain('grad*input', Y, X, [[0, 0, 0]])
                assert any(["DeepExplain detected you are trying" in str(wi.message) for wi in w])

    def test_T_is_tensor(self):
        X = tf.placeholder("float", [None, 3])
        Y = tf.nn.relu(X)
        with DeepExplain(graph=tf.get_default_graph(), session=self.session) as de:
            with self.assertRaises(RuntimeError) as cm:
                de.explain('grad*input', [Y], X, [[0, 0, 0]])
                self.assertIn('T must be a Tensorflow Tensor object',
                              str(cm.exception)
                              )

    def test_X_is_tensor(self):
        X = tf.placeholder("float", [None, 3])
        Y = tf.nn.relu(X)
        with DeepExplain(graph=tf.get_default_graph(), session=self.session) as de:
            with self.assertRaises(RuntimeError) as cm:
                de.explain('grad*input', Y, np.eye(3), [[0, 0, 0]])
                self.assertIn('Tensorflow Tensor object',
                              str(cm.exception)
                              )

    def test_all_in_X_are_tensor(self):
        X = tf.placeholder("float", [None, 3])
        Y = tf.nn.relu(X)
        with DeepExplain(graph=tf.get_default_graph(), session=self.session) as de:
            with self.assertRaises(RuntimeError) as cm:
                de.explain('grad*input', Y, [X, np.eye(3)], [[0, 0, 0]])
                self.assertIn('Tensorflow Tensor object',
                              str(cm.exception)
                              )

    def test_X_has_compatible_batch_dim(self):
        X = tf.placeholder("float", [10, 3])
        Y = tf.nn.relu(X)
        with DeepExplain(graph=tf.get_default_graph(), session=self.session) as de:
            with self.assertRaises(RuntimeError) as cm:
                de.explain('grad*input', Y, X, [[0, 0, 0]], batch_size=2)
                self.assertIn('the first dimension of the input tensor',
                              str(cm.exception)
                              )

    def test_T_has_compatible_batch_dim(self):
        X, out = simpler_model(self.session)
        xi = np.array([[-10, -5]]).repeat(50, 0)
        Y = out * np.expand_dims(np.array(range(50)), -1)
        with DeepExplain(graph=tf.get_default_graph(), session=self.session) as de:
            with self.assertRaises(RuntimeError) as cm:
                de.explain('saliency', Y, X, xi, batch_size=10)
                self.assertIn('the first dimension of the target tensor',
                              str(cm.exception)
                              )


class TestDummyMethod(TestCase):

    def setUp(self):
        self.session = tf.Session()

    def tearDown(self):
        self.session.close()
        tf.reset_default_graph()

    def test_dummy_zero(self):
        with DeepExplain(graph=tf.get_default_graph(), session=self.session) as de:
            X, out = simple_model(tf.nn.sigmoid, self.session)
            xi = np.array([[10, -10]])
            attributions = de.explain('zero', out, X, xi)
            self.assertEqual(attributions.shape, xi.shape)
            np.testing.assert_almost_equal(attributions[0], [0.0, 0.0], 10)

    def test_gradient_restored(self):
        with DeepExplain(graph=tf.get_default_graph(), session=self.session) as de:
            X, out = simple_model(tf.nn.sigmoid, self.session)
            xi = np.array([[10, -10]])
            de.explain('zero', out, X, xi)
            r = train_xor(self.session)
            self.assertTrue(r)


class TestSaliencyMethod(TestCase):

    def setUp(self):
        self.session = tf.Session()

    def tearDown(self):
        self.session.close()
        tf.reset_default_graph()

    def test_saliency_method(self):
        with DeepExplain(graph=tf.get_default_graph(), session=self.session) as de:
            X, out = simpler_model( self.session)
            xi = np.array([[-10, -5], [3, 1]])
            attributions = de.explain('saliency', out, X, xi)
            self.assertEqual(attributions.shape, xi.shape)
            np.testing.assert_almost_equal(attributions, [[0.0, 0.0], [1.0, 1.0]], 10)

    def test_multiple_inputs(self):
        with DeepExplain(graph=tf.get_default_graph(), session=self.session) as de:
            X1, X2, out = simple_multi_inputs_model(self.session)
            xi = [np.array([[-10, -5]]), np.array([[3, 1]])]
            attributions = de.explain('saliency', out, [X1, X2], xi)
            self.assertEqual(len(attributions), len(xi))
            np.testing.assert_almost_equal(attributions[0], [[0.0, 0.0]], 10)
            np.testing.assert_almost_equal(attributions[1], [[2.0, 2.0]], 10)


class TestGradInputMethod(TestCase):

    def setUp(self):
        self.session = tf.Session()

    def tearDown(self):
        self.session.close()
        tf.reset_default_graph()

    def test_saliency_method(self):
        with DeepExplain(graph=tf.get_default_graph(), session=self.session) as de:
            X, out = simpler_model(self.session)
            xi = np.array([[-10, -5], [3, 1]])
            attributions = de.explain('grad*input', out, X, xi)
            self.assertEqual(attributions.shape, xi.shape)
            np.testing.assert_almost_equal(attributions, [[0.0, 0.0], [3.0, -1.0]], 10)

    def test_multiple_inputs(self):
        with DeepExplain(graph=tf.get_default_graph(), session=self.session) as de:
            X1, X2, out = simple_multi_inputs_model(self.session)
            xi = [np.array([[-10, -5]]), np.array([[3, 1]])]
            attributions = de.explain('grad*input', out, [X1, X2], xi)
            self.assertEqual(len(attributions), len(xi))
            np.testing.assert_almost_equal(attributions[0], [[0.0, 0.0]], 10)
            np.testing.assert_almost_equal(attributions[1], [[6.0, 2.0]], 10)


class TestIntegratedGradientsMethod(TestCase):

    def setUp(self):
        self.session = tf.Session()

    def tearDown(self):
        self.session.close()
        tf.reset_default_graph()

    def test_int_grad(self):
        with DeepExplain(graph=tf.get_default_graph(), session=self.session) as de:
            X, out = simpler_model( self.session)
            xi = np.array([[-10, -5], [3, 1]])
            attributions = de.explain('intgrad', out, X, xi)
            self.assertEqual(attributions.shape, xi.shape)
            np.testing.assert_almost_equal(attributions, [[0.0, 0.0], [1.5, -0.5]], 10)

    def test_int_grad_higher_precision(self):
        with DeepExplain(graph=tf.get_default_graph(), session=self.session) as de:
            X, out = simpler_model( self.session)
            xi = np.array([[-10, -5], [3, 1]])
            attributions = de.explain('intgrad', out, X, xi, steps=500)
            self.assertEqual(attributions.shape, xi.shape)
            np.testing.assert_almost_equal(attributions, [[0.0, 0.0], [1.5, -0.5]], 10)

    def test_int_grad_baseline(self):
        with DeepExplain(graph=tf.get_default_graph(), session=self.session) as de:
            X, out = simpler_model(self.session)
            xi = np.array([[2, 0]])
            attributions = de.explain('intgrad', out, X, xi, baseline=np.array([1, 0]))
            self.assertEqual(attributions.shape, xi.shape)
            np.testing.assert_almost_equal(attributions, [[1.0, 0.0]], 10)

    def test_int_grad_baseline_2(self):
        with DeepExplain(graph=tf.get_default_graph(), session=self.session) as de:
            X, out = simpler_model(self.session)
            xi = np.array([[2, 0], [3, 0]])
            attributions = de.explain('intgrad', out, X, xi, baseline=np.array([1, 0]))
            self.assertEqual(attributions.shape, xi.shape)
            np.testing.assert_almost_equal(attributions, [[1.0, 0.0], [2.0, 0.0]], 10)

    def test_multiple_inputs(self):
        with DeepExplain(graph=tf.get_default_graph(), session=self.session) as de:
            X1, X2, out = simple_multi_inputs_model(self.session)
            xi = [np.array([[-10, -5]]), np.array([[3, 1]])]
            attributions = de.explain('intgrad', out, [X1, X2], xi)
            self.assertEqual(len(attributions), len(xi))
            np.testing.assert_almost_equal(attributions[0], [[0.0, 0.0]], 10)
            np.testing.assert_almost_equal(attributions[1], [[6.0, 2.0]], 10)

    def test_multiple_inputs_different_sizes(self):
        with DeepExplain(graph=tf.get_default_graph(), session=self.session) as de:
            X1, X2, out = simple_multi_inputs_model2(self.session)
            xi = [np.array([[-10, -5]]), np.array([[3]])]
            attributions = de.explain('intgrad', out, [X1, X2], xi)
            self.assertEqual(len(attributions), len(xi))
            np.testing.assert_almost_equal(attributions[0], [[0.0, 0.0]], 10)
            np.testing.assert_almost_equal(attributions[1], [[6]], 10)


class TestEpsilonLRPMethod(TestCase):

    def setUp(self):
        self.session = tf.Session()

    def tearDown(self):
        self.session.close()
        tf.reset_default_graph()

    def test_elrp_method(self):
        with DeepExplain(graph=tf.get_default_graph(), session=self.session) as de:
            X, out = simpler_model( self.session)
            xi = np.array([[-10, -5], [3, 1]])
            attributions = de.explain('elrp', out, X, xi)
            self.assertEqual(attributions.shape, xi.shape)
            np.testing.assert_almost_equal(attributions, [[0.0, 0.0], [3.0, -1.0]], 3)

    def test_elrp_epsilon(self):
        with DeepExplain(graph=tf.get_default_graph(), session=self.session) as de:
            X, out = simpler_model( self.session)
            xi = np.array([[-10, -5], [3, 1]])
            attributions = de.explain('elrp', out, X, xi, epsilon=1e-9)
            self.assertEqual(attributions.shape, xi.shape)
            np.testing.assert_almost_equal(attributions, [[0.0, 0.0], [3.0, -1.0]], 7)

    def test_elrp_zero_epsilon(self):
        with DeepExplain(graph=tf.get_default_graph(), session=self.session) as de:
            X, out = simpler_model( self.session)
            xi = np.array([[-10, -5], [3, 1]])
            with self.assertRaises(AssertionError):
                de.explain('elrp', out, X, xi, epsilon=0)

    def test_multiple_inputs(self):
        with DeepExplain(graph=tf.get_default_graph(), session=self.session) as de:
            X1, X2, out = simple_multi_inputs_model(self.session)
            xi = [np.array([[-10, -5]]), np.array([[3, 1]])]
            attributions = de.explain('elrp', out, [X1, X2], xi, epsilon=1e-9)
            self.assertEqual(len(attributions), len(xi))
            np.testing.assert_almost_equal(attributions[0], [[0.0, 0.0]], 7)
            np.testing.assert_almost_equal(attributions[1], [[6.0, 2.0]], 7)



class TestDeepLIFTMethod(TestCase):

    def setUp(self):
        self.session = tf.Session()

    def tearDown(self):
        self.session.close()
        tf.reset_default_graph()

    def test_deeplift(self):
        with DeepExplain(graph=tf.get_default_graph(), session=self.session) as de:
            X, out = simpler_model( self.session)
            xi = np.array([[-10, -5], [3, 1]])
            attributions = de.explain('deeplift', out, X, xi)
            self.assertEqual(attributions.shape, xi.shape)
            np.testing.assert_almost_equal(attributions, [[0.0, 0.0], [2.0, -1.0]], 10)

    def test_deeplift_batches(self):
        with DeepExplain(graph=tf.get_default_graph(), session=self.session) as de:
            X, out = simpler_model( self.session)
            xi = np.array([[-10, -5], [3, 1]])
            xi = np.repeat(xi, 25, 0)
            self.assertEqual(xi.shape[0], 50)
            attributions = de.explain('deeplift', out, X, xi, batch_size=32)
            self.assertEqual(attributions.shape, xi.shape)
            np.testing.assert_almost_equal(attributions, np.repeat([[0.0, 0.0], [2.0, -1.0]], 25, 0), 10)

    def test_deeplift_baseline(self):
        with DeepExplain(graph=tf.get_default_graph(), session=self.session) as de:
            X, out = simpler_model(self.session)
            xi = np.array([[3, 1]])
            attributions = de.explain('deeplift', out, X, xi, baseline=xi[0])
            self.assertEqual(attributions.shape, xi.shape)
            np.testing.assert_almost_equal(attributions, [[0.0, 0.0]], 5)

    def test_multiple_inputs(self):
        with DeepExplain(graph=tf.get_default_graph(), session=self.session) as de:
            X1, X2, out = simple_multi_inputs_model(self.session)
            xi = [np.array([[-10, -5]]), np.array([[3, 1]])]
            attributions = de.explain('deeplift', out, [X1, X2], xi)
            self.assertEqual(len(attributions), len(xi))
            np.testing.assert_almost_equal(attributions[0], [[0.0, 0.0]], 7)
            np.testing.assert_almost_equal(attributions[1], [[6.0, 2.0]], 7)

    def test_multiple_inputs_different_sizes(self):
        with DeepExplain(graph=tf.get_default_graph(), session=self.session) as de:
            X1, X2, out = simple_multi_inputs_model2(self.session)
            xi = [np.array([[-10, -5]]), np.array([[3]])]
            attributions = de.explain('deeplift', out, [X1, X2], xi)
            self.assertEqual(len(attributions), len(xi))
            np.testing.assert_almost_equal(attributions[0], [[0.0, 0.0]], 10)
            np.testing.assert_almost_equal(attributions[1], [[6]], 10)

    def test_multiple_inputs_different_sizes_batches(self):
        with DeepExplain(graph=tf.get_default_graph(), session=self.session) as de:
            X1, X2, out = simple_multi_inputs_model2(self.session)
            xi = [np.array([[-10, -5]]).repeat(50, 0), np.array([[3]]).repeat(50, 0)]
            attributions = de.explain('deeplift', out, [X1, X2], xi, batch_size=32)
            self.assertEqual(len(attributions), len(xi))
            np.testing.assert_almost_equal(attributions[0], np.repeat([[0.0, 0.0]], 50, 0), 10)
            np.testing.assert_almost_equal(attributions[1], np.repeat([[6]], 50, 0), 10)

    def test_multiple_inputs_different_sizes_batches_disable(self):
        with DeepExplain(graph=tf.get_default_graph(), session=self.session) as de:
            X1, X2, out = simple_multi_inputs_model2(self.session)
            xi = [np.array([[-10, -5]]).repeat(50, 0), np.array([[3]]).repeat(50, 0)]
            attributions = de.explain('deeplift', out, [X1, X2], xi, batch_size=None)
            self.assertEqual(len(attributions), len(xi))
            np.testing.assert_almost_equal(attributions[0], np.repeat([[0.0, 0.0]], 50, 0), 10)
            np.testing.assert_almost_equal(attributions[1], np.repeat([[6]], 50, 0), 10)


class TestOcclusionMethod(TestCase):

    def setUp(self):
        self.session = tf.Session()

    def tearDown(self):
        self.session.close()
        tf.reset_default_graph()

    def test_occlusion(self):
        with DeepExplain(graph=tf.get_default_graph(), session=self.session) as de:
            X, out = simpler_model( self.session)
            xi = np.array([[-10, -5], [3, 1]])
            attributions = de.explain('occlusion', out, X, xi)
            self.assertEqual(attributions.shape, xi.shape)
            np.testing.assert_almost_equal(attributions, [[0.0, 0.0], [1.0, -1.0]], 10)

    def test_occlusion_batches(self):
        with DeepExplain(graph=tf.get_default_graph(), session=self.session) as de:
            X, out = simpler_model( self.session)
            xi = np.array([[-10, -5], [3, 1]]).repeat(10, 0)
            attributions = de.explain('occlusion', out, X, xi, batch_size=5)
            self.assertEqual(attributions.shape, xi.shape)
            np.testing.assert_almost_equal(attributions, np.repeat([[0.0, 0.0], [1.0, -1.0]], 10, 0), 10)

    def test_window_shape(self):
        with DeepExplain(graph=tf.get_default_graph(), session=self.session) as de:
            X, out = simpler_model(self.session)
            xi = np.array([[-10, -5], [3, 1]])
            attributions = de.explain('occlusion', out, X, xi, window_shape=(2,))
            self.assertEqual(attributions.shape, xi.shape)
            np.testing.assert_almost_equal(attributions, [[0.0, 0.0], [0.5, 0.5]], 10)

    def test_nan_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with DeepExplain(graph=tf.get_default_graph(), session=self.session) as de:
                X, out = simpler_model(self.session)
                xi = np.array([[-10, -5], [3, 1]])
                attributions = de.explain('occlusion', out, X, xi, step=2)
                self.assertEqual(attributions.shape, xi.shape)
                np.testing.assert_almost_equal(attributions, [[0.0, np.nan], [1.0, np.nan]], 10)
                assert any(["nans" in str(wi.message) for wi in w])

    def test_multiple_inputs_error(self):
        with DeepExplain(graph=tf.get_default_graph(), session=self.session) as de:
            X1, X2, out = simple_multi_inputs_model(self.session)
            xi = [np.array([[-10, -5]]), np.array([[3, 1]])]
            with self.assertRaises(RuntimeError) as cm:
                de.explain('occlusion', out, [X1, X2], xi)
            self.assertIn('not yet supported', str(cm.exception))


class TestShapleySamplingMethod(TestCase):

    def setUp(self):
        self.session = tf.Session()

    def tearDown(self):
        self.session.close()
        tf.reset_default_graph()

    def test_shapley_sampling(self):
        with DeepExplain(graph=tf.get_default_graph(), session=self.session) as de:
            X, out = min_model(self.session)
            xi = np.array([[2, -2], [4, 2]])
            attributions = de.explain('shapley_sampling', out, X, xi, samples=300)
            self.assertEqual(attributions.shape, xi.shape)
            np.testing.assert_almost_equal(attributions, [[0.0, -2.0], [1.0, 1.0]], 1)

    def test_shapley_sampling_batches(self):
        with DeepExplain(graph=tf.get_default_graph(), session=self.session) as de:
            X, out = min_model(self.session)
            xi = np.array([[2, -2], [4, 2]]).repeat(20, 0)
            attributions = de.explain('shapley_sampling', out, X, xi, samples=300, batch_size=5)
            self.assertEqual(attributions.shape, xi.shape)
            np.testing.assert_almost_equal(attributions, np.repeat([[0.0, -2.0], [1.0, 1.0]], 20, 0), 1)

    def test_shapley_sampling_dims(self):
        with DeepExplain(graph=tf.get_default_graph(), session=self.session) as de:
            X, out = min_model_2d(self.session)
            xi = np.array([[[-1, 4], [-2, 1]]])
            attributions = de.explain('shapley_sampling', out, X, xi, samples=300, sampling_dims=[1])
            self.assertEqual(attributions.shape, (1, 2))
            np.testing.assert_almost_equal(attributions, [[-.5, -1.5]], 1)

    def test_multiple_inputs_error(self):
        with DeepExplain(graph=tf.get_default_graph(), session=self.session) as de:
            X1, X2, out = simple_multi_inputs_model(self.session)
            xi = [np.array([[-10, -5]]), np.array([[3, 1]])]
            with self.assertRaises(RuntimeError) as cm:
                de.explain('shapley_sampling', out, [X1, X2], xi)
            self.assertIn('not yet supported', str(cm.exception))

