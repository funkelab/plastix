from plastix.kernels import Parameter
from plastix.kernels.nodes import NodeKernel
import jax.numpy as jnp


class SumNonlinear(NodeKernel):

    bias = Parameter((1,), jnp.zeros)

    def tick(self, edge_states):
        activation = jnp.sum(edge_states) + self.bias
        return jnp.tanh(activation)
