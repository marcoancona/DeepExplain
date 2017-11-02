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
from deepexplain.tensorflow import DeepExplain

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



g = tf.get_default_graph()
for op in g.get_operations():
    # print (op, op.inputs)
    if len(op.inputs) > 0 and not op.name.startswith('gradients'):
        if any(s in op.name for s in ['Relu', 'Sigmoid', 'Tanh', 'Softplus']):
            print (op.name)

print ('==== Gradient override ===')

xi = mnist.test.images[0]
yi = mnist.test.labels[0]

grad = tf.gradients(logits*yi, X)[0]
grads = sess.run(grad, {X: [xi]})
print('This IS the original grad: %f' % np.sum(grads))



with DeepExplain(sess=sess) as de:
    logits2 = neural_net(X)
    explanation = de.explain('grad*input', logits2*yi, X, [xi])
    print('This is the modified gradient: %f' % np.sum(explanation))
    grad = tf.gradients(logits2*yi, X)[0]
    grads = sess.run(grad, {X: [xi]})
    print('This should be original grad: %f' % np.sum(grads))


for op in g.get_operations():
    # print (op, op.inputs)
    if len(op.inputs) > 0 and not op.name.startswith('gradients'):
        if any(s in op.name for s in ['Relu', 'Sigmoid', 'Tanh', 'Softplus']):
            print (op.name)
            print (op.inputs[0])
