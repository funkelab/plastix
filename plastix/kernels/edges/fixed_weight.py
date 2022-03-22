from plastix.kernels import State, Parameter
from plastix.kernels.edges import DirectedEdgeKernel
import jax.numpy as jnp


class FixedWeight(DirectedEdgeKernel):

    signal = State((1,), jnp.zeros)
    weight = Parameter((1,), jnp.ones)

    def tick(self, node):
        self.signal = node.rate * self.weight
