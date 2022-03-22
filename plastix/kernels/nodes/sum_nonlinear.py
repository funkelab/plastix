from plastix.kernels import State, Parameter
from plastix.kernels.nodes import NodeKernel
import jax.numpy as jnp


class SumNonlinear(NodeKernel):

    rate = State((1,), jnp.zeros)
    bias = Parameter((1,), jnp.zeros)

    def tick(self, edge_states):
        activation = jnp.sum(edge_states) + self.bias
        self.rate = jnp.tanh(activation)
