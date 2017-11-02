from unittest import TestCase
import pkg_resources
import logging
import tensorflow as tf
import numpy as np

tf.set_random_seed(10)
np.random.seed(10)

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


def simpler_model(activation, session):
    X = tf.placeholder("float", [None, 2])
    w1 = tf.Variable(initial_value=[[1.0, -1.0], [-1.0, 1.0]])
    b1 = tf.Variable(initial_value=[0.5, -0.5])

    out = activation(tf.matmul(X, w1) + b1)
    session.run(tf.initialize_all_variables())
    return X, out


def train_xor(session):
    tf.set_random_seed(10)
    np.random.seed(10)
    X = tf.placeholder("float", [None, 2])
    Y = tf.placeholder("float", [None, 1])
    w1 = tf.Variable(tf.random_normal([2, 2]))
    b1 = tf.Variable(tf.random_normal([2]))
    w2 = tf.Variable(tf.random_normal([2, 1]))
    b2 = tf.Variable(tf.random_normal([1]))

    l1 = tf.nn.relu(tf.matmul(X, w1) + b1)
    out = tf.nn.relu(tf.matmul(l1, w2) + b2)
    session.run(tf.global_variables_initializer())

    # Define loss and optimizer
    loss = tf.reduce_mean(tf.losses.mean_squared_error(Y, out))
    logging.critical(loss)
    train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

    # Generate dataset random
    x = np.random.randint(0, 2, size=(10, 2))
    y = np.expand_dims(np.logical_or(x[:, 0], x[:, 1]), -1)
    l = None
    for _ in range(100):
        l, _, = session.run([loss, train_step], feed_dict={X: x, Y: y})
    return np.abs(l - 0.1) < 0.01


class TestDeepExplainTF(TestCase):

    def setUp(self):
        logging.critical('Cleanup')
        tf.reset_default_graph()

    def test_tf_available(self):
        try:
            pkg_resources.require('tensorflow>=1.0')
        except Exception:
            self.fail("Tensorflow requirement not met")

    def test_simple_model(self):
        session = tf.Session()
        X, out = simple_model(tf.nn.relu, session)
        xi = np.array([[1, 0]])
        r = session.run(out, {X: xi})
        self.assertEqual(r.shape, xi.shape)
        np.testing.assert_equal(r[0], [2.75,  5.5])

    def test_training(self):
        session = tf.Session()
        r = train_xor(session)
        self.assertTrue(r)

    def test_dummy_zero(self):
        pass

    def test_saliency_maps(self):
        pass

    def test_gradient_x_input(self):
        pass

    def test_deep_lift(self):
        pass

    def test_elrp(self):
        pass

    def test_integrated_gradients(self):
        pass