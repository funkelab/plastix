import jax.numpy as jnp
import plastix as px
import unittest
import jax


class TestSparseLayer(unittest.TestCase):
    def test_and(self):

        n = 2
        m = 1
        # dense layer as sparse layer
        edges = [(0, 0), (1, 0)]

        layer = px.layers.SparseLayer(
            n,
            m,
            edges,
            px.kernels.edges.FixedWeight(),
            px.kernels.nodes.SumNonlinear(),
        )

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
        edges = [(42, 99), (23, 99)]

        layer = px.layers.SparseLayer(
            n,
            m,
            edges,
            px.kernels.edges.FixedWeight(),
            px.kernels.nodes.SumNonlinear(),
        )

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

    def test_sparse_dense(self):
        dense_layer = px.layers.DenseLayer(
            5,
            3,
            px.kernels.edges.FixedWeight(),
            px.kernels.nodes.SumNonlinear(),
        )
        edges = [(i, j) for j in range(3) for i in range(5)]
        sparse_layer = px.layers.SparseLayer(
            5,
            3,
            edges,
            px.kernels.edges.FixedWeight(),
            px.kernels.nodes.SumNonlinear(),
        )
        dl_state = dense_layer.init_state()
        dl_parameters = dense_layer.init_parameters()
        sl_state = sparse_layer.init_state()
        sl_parameters = sparse_layer.init_parameters()

        dl_parameters.edges.weight *= 0.5
        sl_parameters.edges.weight *= 0.5

        # 0, 0, 0, 0, 0 -> 0, 0, 0
        layer_input = jnp.array([[0.0] for _ in range(5)])
        dl_state.input_nodes.rate = layer_input
        sl_state.input_nodes.rate = layer_input

        dl_state = dense_layer.update_state(dl_state, dl_parameters)
        sl_state = sparse_layer.update_state(sl_state, sl_parameters)
        assert (dl_state.output_nodes.rate == sl_state.output_nodes.rate).all()

        # 1, 1, 0, 0, 0 -> 0.96, 0.96, 0.96
        layer_input = jnp.array([[0.0] for _ in range(5)])
        dl_state.input_nodes.rate = layer_input
        sl_state.input_nodes.rate = layer_input

        dl_state = dense_layer.update_state(dl_state, dl_parameters)
        sl_state = sparse_layer.update_state(sl_state, sl_parameters)
        assert (dl_state.output_nodes.rate == sl_state.output_nodes.rate).all()

        # random input to output
        key = jax.random.PRNGKey(seed=0)
        layer_input = jax.random.normal(key, (5,))
        dl_state.input_nodes.rate = layer_input
        sl_state.input_nodes.rate = layer_input

        dl_state = dense_layer.update_state(dl_state, dl_parameters)
        sl_state = sparse_layer.update_state(sl_state, sl_parameters)
        assert (
            round(dl_state.output_nodes.rate, 5)
            == round(sl_state.output_nodes.rate, 5)
        ).all()
