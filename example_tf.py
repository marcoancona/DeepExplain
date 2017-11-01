"""
A  simple MNIST classifier.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import ops


import numpy as np
import tensorflow as tf


tf.set_random_seed(10)

# Parameters
learning_rate = 0.001
num_steps = 1
batch_size = 128
display_step = 100

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)


# Import data
mnist = input_data.read_data_sets('tmp/mnist', one_hot=True)

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}


# Create model
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Construct model
logits = neural_net(X)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

# Evaluate model
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Train
for step in range(num_steps):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)

    sess.run(train_step, feed_dict={X: batch_xs, Y: batch_ys})
    if step % display_step == 0 or step == 1:
        # Calculate batch loss and accuracy
        acc = sess.run(accuracy, feed_dict={X: batch_xs, Y: batch_ys})
        print("Step " + str(step) + ", Minibatch Acc= " + \
              "{:.3f}".format(acc))

# Test trained model
print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))

"""
Exmplain network prediction
"""



@ops.RegisterGradient("MyGradientDEEPLIFT")
def MyGradientDEEPLIFT(op, grad):
    input = op.inputs[0]
    return tf.zeros_like(input)

@ops.RegisterGradient("MyGradientGRAD")
def MyGradientGRAD(op, grad):
    input = op.inputs[0]
    return tf.ones_like(input)

from contextlib import contextmanager

class DeepExplain(object):

    # @staticmethod
    # @contextmanager
    # def use(self, method):
    #     with self.graph.as_default():
    #         with self.graph.gradient_override_map({'Relu': 'MyStopGradient'}):
    #             print('Override')
    #             yield
    #             return self

    def __init__(self, method, graph=tf.get_default_graph(), sess=tf.Session()):
        self.method = method
        self.graph = graph
        self.session = sess
        self.graph_context = self.graph.as_default()
        self.override_context = self.graph.gradient_override_map(self.get_override_map())

    def get_override_map(self):
        if self.method == '' or self.method == None:
            return {}
        else:
            return {'Relu': 'MyGradient'+self.method.upper()}

    def __enter__(self):
        #Override gradient of all ops created here
        self.graph_context.__enter__()
        self.override_context.__enter__()
        print ('Override')
        return self

    def __exit__(self, type, value, traceback):
        self.graph_context.__exit__(type, value, traceback)
        self.override_context.__exit__(type, value, traceback)

    def explain(self, T, X, xs):
        compute_grad = tf.gradients(T, X)[0]
        explanation = sess.run(compute_grad, {X: xs})
        print ('Oh, its ' + self.method)
        return explanation







g = tf.get_default_graph()
for op in g.get_operations():
    # print (op, op.inputs)
    if len(op.inputs) > 0 and not op.name.startswith('gradients'):
        if any(s in op.name for s in ['Relu', 'Sigmoid', 'Tanh', 'Softplus']):
            print (op.name)

print ('==== Gradient override ===')

xi = mnist.test.images[0]
yi = mnist.test.labels[0]



with DeepExplain('deeplift') as explanator:
    print (explanator)
    logits2 = neural_net(X)
    explanation = explanator.explain(logits2*yi, X, [xi])
    print(np.sum(explanation))


for op in g.get_operations():
    # print (op, op.inputs)
    if len(op.inputs) > 0 and not op.name.startswith('gradients'):
        if any(s in op.name for s in ['Relu', 'Sigmoid', 'Tanh', 'Softplus']):
            print (op.name)
            print (op.inputs[0])
