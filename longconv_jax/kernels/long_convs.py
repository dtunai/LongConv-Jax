import jax
import dataclasses
import numpy as np
import jax.numpy as jnp
from flax import linen as nn
from typing import Callable, Any, Dict
from flax.core import freeze, unfreeze
from jax.nn.initializers import normal as normal_init


class LongConv(nn.Module):
    """
    Flax Linen module for LongConv, handles sequences.

    Attr:
        H (int): The dimension of the latent space
        L (int): The sequence length to be processed
        channels (int): The number of channels in the input data. Default is 2
        dropout_rate (float): Dropout rate applied to the output of the convolution operation. Default is 0.1
        kernel_learning_rate (float): Learning rate for the kernel. If None, uses the default learning rate. Default is None
        kernel_lam (float): Regularization term for the kernel weights. Default is 0.1
        kernel_dropout_rate (float): Dropout rate applied to the kernel weights. Default is 0

    Methods:
        __call__(u, deterministic=True): Applies the long convolution computations on the input tensor `u`
    """

    H: int
    L: int
    channels: int = 2
    dropout_rate: float = 0.1
    kernel_learning_rate: float = None
    kernel_lam: float = 0.1
    kernel_dropout_rate: float = 0

    @nn.compact
    def __call__(self, u, deterministic=True):
        L = u.shape[-1]
        D = self.param("D", nn.initializers.normal(), (self.channels, self.H))
        kernel = self.param(
            "kernel", nn.initializers.normal(stddev=0.002), (self.channels, self.H, self.L * 2)
        )

        k = jax.nn.relu(jnp.abs(kernel) - self.kernel_lam) * jnp.sign(kernel)
        k = nn.Dropout(rate=self.kernel_dropout_rate)(k, deterministic=deterministic)
        k_f = jnp.fft.rfft(k, n=2 * L)
        u_f = jnp.fft.rfft(u, n=2 * L)
        y_f = jnp.einsum("bhl,chl->bchl", u_f, k_f)
        y = jnp.fft.irfft(y_f, n=2 * L)[..., :L]
        y += jnp.einsum("bhl,ch->bchl", u, D)
        y = y.reshape(*y.shape[:-3], -1, y.shape[-1])
        y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=deterministic)
        y = nn.gelu(y)
        y = y.transpose((0, 2, 1))
        y = nn.Dense(features=2 * self.H)(y)
        y = nn.glu(y)
        y = y.transpose((0, 2, 1))

        return y


class LongConvModel(nn.Module):
    """
    Flax Linen model that incorporates LongConv modules for processing long convolution sequences

    Attr:
        d_input (int): Dimensionality of the input data
        d_output (int): Dimensionality of the output data. Default is 10
        d_model (int): Dimensionality of the model. Default is 512
        n_layers (int): Number of LongConv layers in the model. Default is 6
        dropout (float): Dropout rate applied throughout the model. Default is 0.1
        prenorm (bool): Whether to use pre-normalization. Default is False
        conv_kwargs (Dict[str, Any]): Additional keyword arguments to be passed to the LongConv layers

    Methods:
        __call__(x, deterministic=True): Processes the input tensor `x` through the model
    """

    d_input: int
    d_output: int = 10
    d_model: int = 512
    n_layers: int = 6
    dropout: float = 0.1
    prenorm: bool = False
    conv_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def setup(self):
        self.encoder = nn.Dense(self.d_model)
        self.conv_layers = [
            LongConv(self.d_model, L=1024, dropout_rate=self.dropout, **self.conv_kwargs)
            for _ in range(self.n_layers)
        ]
        self.norms = [nn.LayerNorm(epsilon=1e-6) for _ in range(self.n_layers)]
        self.dropout_layer = nn.Dropout(rate=self.dropout)
        self.decoder = nn.Dense(self.d_output)
        self.kernel_learning_rate = self.conv_kwargs.get("kernel_learning_rate", None)

    def __call__(self, x, deterministic=True):
        x = self.encoder(x)
        x = jnp.transpose(x, (0, 2, 1))
        for i in range(self.n_layers):
            z = x
            if self.prenorm:
                z = self.norms[i](z)
            z = self.conv_layers[i](z, deterministic=deterministic)
            z = self.dropout_layer(z, deterministic=deterministic)
            x = z + x
            if not self.prenorm:
                x = self.norms[i](x)

        x = jnp.transpose(x, (0, 2, 1))
        x = jnp.mean(x, axis=1)
        x = self.decoder(x)
        return x
