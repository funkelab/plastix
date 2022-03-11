from .node_kernel import NodeKernel
import jax.numpy as jnp


class SumNonlinear(NodeKernel):

    default_parameters = jnp.array([0.0])

    def tick(self, edge_states, parameters):
        activation = jnp.sum(edge_states) + parameters
        return jnp.tanh(activation)
