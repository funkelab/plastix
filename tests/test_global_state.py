import plastix as px
import unittest


class TestGlobalStates(unittest.TestCase):

    def __test_global(self):

        layer = px.layers.DenseLayer(
            2, 1,
            px.kernels.edges.FixedWeight(),
            px.kernels.nodes.SumNonlinear())

        layer.edge_parameters = jnp.array([[0.5], [0.5]])

        # 0, 0 -> 0
        input_node_states = jnp.array([[0.0], [0.0]])
        output_node_states = layer(input_node_states)

