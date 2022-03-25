from plastix.kernels import State, Parameter
from plastix.kernels.edges import EdgeKernel
import jax.numpy as jnp


class FixedWeight(EdgeKernel):

    signal = State((1,), jnp.zeros)
    weight = Parameter((1,), jnp.ones)

    def update_state(self, input_node):
        self.signal = input_node.rate * self.weight

    def update_parameters(self, input_node, output_node):
        pass
