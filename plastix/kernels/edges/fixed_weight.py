from .directed_edge_kernel import DirectedEdgeKernel
import jax.numpy as jnp


class FixedWeight(DirectedEdgeKernel):

    default_parameters = jnp.array([1.0])

    def tick(self, node_state, parameters):
        return node_state * parameters
