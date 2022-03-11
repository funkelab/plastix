import jax.numpy as jnp
import plastix as px
import unittest


class TestDenseLayer(unittest.TestCase):

    def test_and(self):

        layer = px.layers.DenseLayer(
            2, 1,
            px.kernels.edges.FixedWeight(),
            px.kernels.nodes.SumNonlinear())

        layer.edge_parameters = jnp.array([[0.5], [0.5]])

        # 0, 0 -> 0

        input_node_states = jnp.array([[0.0], [0.0]])
        output_node_states = layer(input_node_states)
        assert output_node_states[0] == 0

        # 0, 1 -> <0.5

        input_node_states = jnp.array([[0.0], [1.0]])
        output_node_states = layer(input_node_states)
        assert output_node_states[0] <= 0.5

        # 1, 0 -> <0.5

        input_node_states = jnp.array([[1.0], [0.0]])
        output_node_states = layer(input_node_states)
        assert output_node_states[0] <= 0.5

        # 1, 1 -> >0.5

        input_node_states = jnp.array([[1.0], [1.0]])
        output_node_states = layer(input_node_states)
        assert output_node_states[0] >= 0.5
