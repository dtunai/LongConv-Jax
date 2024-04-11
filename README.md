## Long Convolutions (Jax / Flax)

This package provides Flax Linen modules for the paper ["Simple Long Convolutions for Sequence Modeling"](https://hazyresearch.stanford.edu/blog/2023-02-15-long-convs) from HazyResearch. Uses FFT convolution to compute a long convolution in O(N log N) time (instead of O(N^^2)), and applies a simple regularization through a Squash operator to the kernel weights. LongConv is particularly effective for processing long convolution sequences. It includes modules for individual LongConv layers as well as a model that incorporates multiple LongConv layers for sequence processing tasks.

## Getting Started

**Requirements**

```bash
jaxlib==0.4.25
jax==0.4.25
numpy==1.25.2
flax==0.8.1
```

You can install the package using `pip3 install -e .`:

```bash
git clone https://github.com/attophyd/LongConv-Jax
cd LongConv-Jax
pip3 install -e .
```

## Usage

Instantiate the model:

```python
model = LongConvModel(d_input=..., d_output=..., d_model=..., n_layers=..., dropout=..., prenorm=..., conv_kwargs={...})
```

or modify the `usage.py`.

## License

This package is licensed under the Apache License - see the LICENSE file for details.