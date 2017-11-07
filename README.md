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
- **Occlusion**, as an extension of the [grey-box method by Zeiler *et al*](https://arxiv.org/abs/1311.2901).

## Installation
```unix

```