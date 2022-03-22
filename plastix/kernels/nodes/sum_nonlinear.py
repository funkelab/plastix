from plastix.kernels import State, Parameter
from plastix.kernels.nodes import NodeKernel
import jax.numpy as jnp


class SumNonlinear(NodeKernel):

    rate = State((1,), jnp.zeros)
    bias = Parameter((1,), jnp.zeros)

    def tick(self, edges):
        activation = jnp.sum(edges.signal) + self.bias
        self.rate = jnp.tanh(activation)
