from plastix.kernels import State, Parameter
from plastix.kernels.edges import DirectedEdgeKernel
import jax.numpy as jnp


class FixedWeight(DirectedEdgeKernel):

    signal = State((1,), jnp.zeros)
    weight = Parameter((1,), jnp.ones)

    def tick(self, node_state):
        self.signal = node_state * self.weight
