import jax.numpy as jnp
import plastix as px
import unittest


class TestDenseLayer(unittest.TestCase):

    def test_and(self):

        layer = px.layers.DenseLayer(
            2, 1,
            px.kernels.edges.FixedWeight(),
            px.kernels.nodes.SumNonlinear())

        state = layer.init_state()
        parameters = layer.init_parameters()
        parameters.edges.weight *= 0.5

        # 0, 0 -> 0

        state.input_nodes.rate = jnp.array([[0.0], [0.0]])
        state = layer.update_state(state, parameters)
        assert state.output_nodes.rate[0] == 0

        # 0, 1 -> <0.5

        state.input_nodes.rate = jnp.array([[0.0], [1.0]])
        state = layer.update_state(state, parameters)
        assert state.output_nodes.rate[0] < 0.5

        # 1, 0 -> <0.5

        state.input_nodes.rate = jnp.array([[1.0], [0.0]])
        state = layer.update_state(state, parameters)
        assert state.output_nodes.rate[0] < 0.5

        # 1, 1 -> >0.5

        state.input_nodes.rate = jnp.array([[1.0], [1.0]])
        state = layer.update_state(state, parameters)
        assert state.output_nodes.rate[0] > 0.5
