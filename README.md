DeepExplain: attribution methods for Deep Neural Networks
===
DeepExplain provides a unified framework for state-of-the-art gradient *and* perturbation-based attribution methods.
It can be used by practitioners to better undertand the behavior of existing models, as well as by researches for
benchmarking.

It supports **Tensorflow** as well as **Keras** with Tensorflow backend. Support for PyTorch is planned.

Implements the following methods:

**Gradient-based attribution methods**
- [**Saliency maps**](https://arxiv.org/abs/1312.6034)
- [**Gradient * Input**](https://arxiv.org/abs/1605.01713)
- [**Integrated Gradients**](https://arxiv.org/abs/1703.01365)
- [**DeepLIFT**](https://arxiv.org/abs/1704.02685), in its first variant with Rescale rule (*)
- [**ε-LRP**](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140) (*)

Methods marked with (*) are implemented as modified chain-rule, as better explained in []()

**Pertubration-based attribution methods**
- [**Occlusion**](https://arxiv.org/abs/1311.2901), as an extension
of the [grey-box method by Zeiler *et al*](https://arxiv.org/abs/1311.2901).


## Installation
```unix
pip install -e git+https://github.com/marcoancona/DeepExplain.git
```

Notice that DeepExplain assumes you already have installed `Tensorflow > 1.0` and (optionally) `Keras > 2.0`.

## Quick start

Working examples for Tensorflow and Keras can be found in the `example` folder of the repository. DeepExplain
consists of a single method: `explain(method_name, target_tensor, input_tensor, samples, ...args)`.


Parameter name | Type | Description
---------------|------|------------
`method_name` | string, required | Name of the method to run
`target_tensor` | Tensor, required | Tensorflow Tensor object representing the output of the model for which attributions are seeked. See below for how to select a good target tensor.
`input_tensor` | Tensor, required | Tensorflow Placeholder object, used as input to the network.
`samples` | numpy array, required | Batch of input samples to be fed to `input_tensor` and for which attributions are seeked. Notice that the first dimension must always be the batch size.
`...args` | various, optional | Method-specific parameters (see below).

The method `explain` must be called within a DeepExplain context:

```python
from deepexplain.tensorflow import DeepExplain

# Option 1. Create and train your model within a DeepExplain context

with DeepExplain(...) as de:  # < enter DeepExplain context
    model = init_model()  # < construct the model
    model.fit()           # < train the model

    attributions = de.explain(...)  # < compute attributions

# Option 2. First create and train your model, then apply DeepExplain.
# IMPORTANT: in order to work correctly, the graph to analyze
# must always be (re)constructed within the context!

model = init_model()  # < construct the model
model.fit()           # < train the model

with DeepExplain(...) as de:  # < enter DeepExplain context
    new_model = init_model()  # < assumes init_model() returns a *new* model with the weights of `model`
    attributions = de.explain(...)  # < compute attributions
```
