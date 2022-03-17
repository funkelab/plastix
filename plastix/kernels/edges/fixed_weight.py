from plastix.kernels import Parameter
from plastix.kernels.edges import DirectedEdgeKernel
import jax.numpy as jnp


class FixedWeight(DirectedEdgeKernel):

    weight = Parameter((1,), jnp.ones)

    def tick(self, node_state):
        return node_state * self.weight
