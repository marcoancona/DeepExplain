DeepExplain: attribution methods for Deep Learning
[![Build Status](https://travis-ci.org/marcoancona/DeepExplain.svg?branch=master)](https://travis-ci.org/marcoancona/DeepExplain)
===
DeepExplain provides a unified framework for state-of-the-art gradient *and* perturbation-based attribution methods.
It can be used by researchers and practitioners for better undertanding the recommended existing models, as well for benchmarking other attribution methods.

It supports **Tensorflow** as well as **Keras** with Tensorflow backend. Support for PyTorch is planned.

Implements the following methods:

**Gradient-based attribution methods**
- [**Saliency maps**](https://arxiv.org/abs/1312.6034)
- [**Gradient * Input**](https://arxiv.org/abs/1605.01713)
- [**Integrated Gradients**](https://arxiv.org/abs/1703.01365)
- [**DeepLIFT**](https://arxiv.org/abs/1704.02685), in its first variant with Rescale rule (*)
- [**ε-LRP**](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140) (*)

Methods marked with (*) are implemented as modified chain-rule, as better explained in [Towards better understanding of gradient-based attribution methods for Deep Neural Networks](https://openreview.net/forum?id=Sy21R9JAW), Ancona *et al*, ICLR 2018. As such, the result might be slightly different from the original implementation.

**Pertubration-based attribution methods**
- [**Occlusion**](https://arxiv.org/abs/1311.2901), as an extension
of the [grey-box method by Zeiler *et al*](https://arxiv.org/abs/1311.2901).
- [**Shapley Value sampling**](https://www.sciencedirect.com/science/article/pii/S0305054808000804)

## What are attributions?
Consider a network and a specific input to this network (eg. an image, if the network is trained for image classification). The input is multi-dimensional, made of several features. In the case of images, each pixel can be considered a feature. The goal of an attribution method is to determine a real value `R(x_i)` for each input feature, with respect to a target neuron of interest (for example, the activation of the neuron corresponsing to the correct class). 

When the attributions of all input features are arranged together to have the same shape of the input sample we talk about *attribution maps* (as in the picture below), where red and blue colors indicate respectively features that contribute positively to the activation of the target output and features having a suppressing effect on it.
![Attribution methods comparison on InceptionV3](https://github.com/marcoancona/DeepExplain/blob/master/docs/comparison.png)

This can help to better understand the network behavior, which features mostly contribute to the output and possible reasons for missclassification.


DeepExplain Quickstart
===
## Installation
```unix
pip install -e git+https://github.com/marcoancona/DeepExplain.git#egg=deepexplain
```

Notice that DeepExplain assumes you already have installed `Tensorflow > 1.0` and (optionally) `Keras > 2.0`.

## Usage

Working examples for Tensorflow and Keras can be found in the `example` folder of the repository. DeepExplain
consists of a single method: `explain(method_name, target_tensor, input_tensor, samples, ...args)`.


Parameter name | Short name | Type | Description
---------------|------|------|------------
`method_name` | | string, required | Name of the method to run (see [Which method to use?](#which-method-to-use)).
`target_tensor` | `T` | Tensor, required | Tensorflow Tensor object representing the output of the model for which attributions are seeked (see [Which tensor to target?](#which-neuron-to-target)).
`input_tensor` | `X` | Tensor, required | Symbolic input to the network.
`input_data` | `xs` | numpy array, required | Batch of input samples to be fed to `X` and for which attributions are seeked. Notice that the first dimension must always be the batch size.
`target_weights` | `ys` | numpy array, optional | Batch of weights to be applied to `T` if this has more than one output. Usually necessary on classification problems where there are multiple output units and we need to target a specific one to generate explanations for. In this case, `ys` can be provided with the one-hot encoding of the desired unit.
`batch_size` | |int, optional| By default, DeepExplain will try to evaluate the model using all data in `xs` at the same time. If `xs` contains many samples, it might be necessary to split the processing in batches. In this case, providing a `batch_size` greater than zero will automatically split the evaluation into chunks of the given size.
`...args` | | various, optional | Method-specific parameters (see below).

The method `explain` must be called within a DeepExplain context:

```python
# Pseudo-code
from deepexplain.tensorflow import DeepExplain

# Option 1. Create and train your model within a DeepExplain context

with DeepExplain(session=...) as de:  # < enter DeepExplain context
    model = init_model()  # < construct the model
    model.fit()           # < train the model

    attributions = de.explain(...)  # < compute attributions

# Option 2. First create and train your model, then apply DeepExplain.
# IMPORTANT: in order to work correctly, the graph to analyze
# must always be (re)constructed within the context!

model = init_model()  # < construct the model
model.fit()           # < train the model

with DeepExplain(session=...) as de:  # < enter DeepExplain context
    new_model = init_model()  # < assumes init_model() returns a *new* model with the weights of `model`
    attributions = de.explain(...)  # < compute attributions
```

When initializing the context, make sure to pass the `session` parameter:

```python
# With Tensorflow
import tensorflow as tf
# ...build model
sess = tf.Session()
# ... use session to train your model if necessary
with DeepExplain(session=sess) as de:
    ...

# With Keras
import keras
from keras import backend as K

model = Sequential()  # functional API is also supported
# ... build model and train

with DeepExplain(session=K.get_session()) as de:
    ...
```

See concrete examples [here](https://github.com/marcoancona/DeepExplain/tree/master/examples).

## Which method to use?
DeepExplain supports several methods. The main partition is between *gradient-based methods* and *perturbation-based methods*. The former are faster, given that they estimate attributions with a few forward and backward iterations through the network. The latter perturb the input and measure the change in output with respect to the original input. This requires to sequentially test each feature (or group of features) and therefore takes more time, but tends to produce smoother results.

Cooperative game theory suggests [**Shapley Values**](https://en.wikipedia.org/wiki/Shapley_value) as a unique way to distribute attribution to features such that some important theoretical properties are satisfied. Unfortunately, computing Shapley Values exactly is prohibitively expensive, therefore DeepExplain provides a sampling-based approximation. By changing the `samples` parameters, one can adjust the trade-off between performance and error. Notice that this method will still be significantly slower than other methods in this library.

Some methods allow tunable parameters. See the table below.

Method | `method_name` | Optional parameters | Notes
---------------|:------|:------------|-----
Saliency | `saliency` |  | [*Gradient*] Only positive attributions.
Gradient * Input | `grad*input` |  | [*Gradient*] Fast. May be affected by noisy gradients and saturation of the nonlinerities.
Integrated Gradients | `intgrad` |`steps`, `baseline` | [*Gradient*] Similar to Gradient * Input, but performs `steps` iterations (default: 100) though the network, varying the input from `baseline` (default: zero) to the actual provided sample. When provided, `baseline` must be a numpy array with the size of the input (but no batch dimension since the same baseline will be used for all inputs in the batch).
epsilon-LRP | `elrp` | `epsilon` | [*Gradient*]Computes Layer-wise Relevance Propagation. Only recommanded with ReLU or Tanh nonlinearities. Value for `epsilon` must be greater than zero (default: .0001).
DeepLIFT (Rescale) | `deeplift` | `baseline` |  [*Gradient*] In most cases a faster approximation of Integrated Gradients. Do not apply to networks with multiplicative units (ie. LSTM or GRU). When provided, `baseline` must be a numpy array with the size of the input, without the batch dimension (default: zero).
Occlusion | `occlusion` | `window_shape`, `step` | [*Perturbation*] Computes rolling window view of the input array and replace each window with zero values, measuring the effect of the perturbation on the target output. The optional parameters `window_shape` and `step` behave like in [skimage](http://scikit-image.org/docs/dev/api/skimage.util.html#skimage.util.view_as_windows). By default, each feature is tested independently (`window_shape=1` and `step=1`), however this might be extremely slow for large inputs (such as ImageNet images). When the input presents some local coherence (eg. images), you might prefer larger values for `window_shape`. In this case the attributions of the features in each window will be summed up. Notice that the result might vary significantly for different window sizes.
Shapley Value sampling | `shapley_sampling` | `samples`, `sampling_dims` | [*Perturbation*] Computes approximate Shapley Values by sampling `samples` times each input feature. Notice that this method can be significantly slower than all the others as it runs the network `samples*n` times, where `n` is the number of input features in your input. The parameter `sampling_dims` (a list of integers) can be used to select which dimensions should be sampled. For example, if the inputs are RGB images, `sampling_dims=[1,2]` would sample pixels considering the three color channels atomic. Instead `sampling_dims=[1,2,3]` (default) will samples over the channels as well.

## Which neuron to target?
In general, any tensor that represents the activation of any hidden or output neuron can be user as `target_tensor`. If your network performs a classification task (ie. one output neuron for each possible class) you might want to target the neuron corresponding to the *correct class* for a given sample, such that the attribution map might help you undertand the reasons for this neuron to (not) activate. However you can also target the activation of another class, for example a class that is often missclassified, to have insight about features that activate this class.

**Important**: Tensors in Tensorflow and Keras usually include the activations of *all* neurons of a layer. If you pass such a tensor to `explain` you will get the *sum* attribution map for all neurons the Tensor refers to. If you want to target a specific neuron you need either to slice the component you are interested in or multiply it for a binary mask that only select the target neuron.

```python
# Example on MNIST (classification, with 10 output classes)
X = Placeholder(...)  # input tensor
T = model(X) # output layer, 2-dimensional Tensor of shape (1, 10), where first dimension is the batch size
ys = [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]]  # numpy array of shape (1, 10) with one-hot encoding of labels

# We need to target only one of the 10 output units in `T`
# Option 1 (recommanded): use the `ys` parameter
de.explain('method_name', T, X, xs, ys=ys)

# Option 2: manually mask the target. This will not work with batch processing.
T *=  ys # < masked target tensor: only the second component of `logits` will be used to compute attributions
de.explain('method_name', T, X, xs)

```

**Softmax**: if the network last activation is a Softmax, it is recommanded to target the activations *before* this normalization. 

### Performance: Explainer API
If you need to run `explain()` multiple times (for example, new data to process with the same model comes in over time) it is recommanded that you use the Explainer API. This provides a way to *compile* the graph operations needed to generate the explanations and *evaluate* this graph in two different steps. 

Within a DeepExplain context (`de`), call `de.get_explainer()`. This method takes the same arguments of `explain()` except `xs`, `ys` and `batch_size`. It returns an explainer object (`explainer`) which provides a `run()` method. Call `explainer.run(xs, [ys], [batch_size])` to generate the explanations. Calling `run()` multiple times will not add new operations to the computational graph.


```python
# Normal API: 

for i in range(100):
    # The following line will become slower and slower as new operations are added to the computational graph at each iteration
    attributions = de.explain('saliency', T, X, xs[i], ys=ys[i], batch_size=3)
    
 # Use the Explainer API instead:
 
 # First create an explainer
 explainer = de.get_explainer('saliency', T, X)
 for i in range(100):
    # Then generate explanations for some data without slowing things down
    attributions = explainer.run(xs[i], ys=ys[i], batch_size=3)  
 ```


### NLP / Embedding lookups
The most common cause of `ValueError("None values not supported.")` is `run()` being called with a `tensor_input` and `target_tensor` that are disconnected in the backpropagation. This is common when an embedding lookup layer is used, since the lookup operation does not propagate the gradient. To generate attributions for NLP models, the input of DeepExplain should be the result of the embedding lookup instead of the original model input. Then, attributions for each word are found by summing up along the appropriate dimension of the resulting attribution matrix.

Tensorflow pseudocode:
```python
input_x = graph.get_operation_by_name("input_x").outputs[0]
# Get a reference to the embedding tensor
embedding = graph.get_operation_by_name("embedding").outputs[0]
pre_softmax = graph.get_operation_by_name("output/scores").outputs[0]

# Evaluate the embedding tensor on the model input (in other words, perform the lookup)
embedding_out = sess.run(embedding, {input_x: x_test})
# Run DeepExplain with the embedding as input
attributions = de.explain('elrp', pre_softmax * y_test_logits, embedding, embedding_out)
```

### Multiple inputs
Models with multiple inputs are supported for gradient-based methods. Instead, the `Occlusion` method will raise an exception if
 called on a model with multiple inputs (how perturbation should be generated for multiple inputs is actually not well defined).

For a minimal (toy) example see the [example folder](https://github.com/marcoancona/DeepExplain/tree/master/examples).

## Contributing
DeepExplain is still in active development. If you experience problems, feel free to open an issue. Contributions to extend the functinalities of this framework and/or to add support for other methods are welcome. 

## License
MIT
