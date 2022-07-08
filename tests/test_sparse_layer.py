import jax.numpy as jnp
import plastix as px
import unittest


class TestSparseLayer(unittest.TestCase):

    def test_and(self):

        n = 2
        m = 1
        # dense layer as sparse layer
        edges = [
            (0, 0),
            (1, 0)
        ]

        layer = px.layers.SparseLayer(
            n, m,
            edges,
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

    def test_sparse_and(self):

        n = 200
        m = 100
        # dense layer as sparse layer
        edges = [
            (42, 99),
            (23, 99)
        ]

        layer = px.layers.SparseLayer(
            n, m,
            edges,
            px.kernels.edges.FixedWeight(),
            px.kernels.nodes.SumNonlinear())

        state = layer.init_state()
        parameters = layer.init_parameters()
        parameters.edges.weight *= 0.5

        # 0, 0 -> 0
        inputs = [[0.0] for _ in range(n)]

        state.input_nodes.rate = jnp.array(inputs)
        state = layer.update_state(state, parameters)
        assert state.output_nodes.rate[99] == 0

        # 0, 1 -> <0.5
        inputs[23] = [1.0]

        state.input_nodes.rate = jnp.array(inputs)
        state = layer.update_state(state, parameters)
        assert state.output_nodes.rate[99] < 0.5

        # 1, 0 -> <0.5
        inputs[23] = [0.0]
        inputs[42] = [1.0]

        state.input_nodes.rate = jnp.array(inputs)
        state = layer.update_state(state, parameters)
        assert state.output_nodes.rate[99] < 0.5

        # 1, 1 -> >0.5
        inputs[23] = [1.0]

        state.input_nodes.rate = jnp.array(inputs)
        state = layer.update_state(state, parameters)
        assert state.output_nodes.rate[99] > 0.5
