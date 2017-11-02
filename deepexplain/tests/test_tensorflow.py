from unittest import TestCase
import pkg_resources
import logging
import tensorflow as tf
import numpy as np


def tf_simple_model(activation, session):
    X = tf.placeholder("float", [None, 2])
    w1 = tf.Variable(initial_value=[[1.0, 1.0], [1.0, 1.0]])
    b1 = tf.Variable(initial_value=[0.0, 0.0])
    w2 = tf.Variable(initial_value=[[1.0, 1.0], [1.0, 1.0]])
    b2 = tf.Variable(initial_value=[0.0, 0.0])

    layer1 = activation(tf.matmul(X, w1) + b1)
    out = tf.matmul(layer1, w2) + b2
    session.run(tf.initialize_all_variables())
    return X, out


class TestDeepExplainTF(TestCase):

    def setUp(self):
        logging.critical('Cleanup')
        tf.reset_default_graph()

    def test_tf_available(self):
        try:
            pkg_resources.require('tensorflow>=1.0')
        except Exception:
            self.fail("Tensorflow requirement not met")

    def test_simple_tf_model(self):
        session = tf.Session()
        X, out = tf_simple_model(tf.nn.relu, session)
        xi = np.array([[1,1]])
        r = session.run(out, {X: xi})
        self.assertEqual(r.shape, xi.shape)
        np.testing.assert_equal(r[0], [4.0, 4.0])

