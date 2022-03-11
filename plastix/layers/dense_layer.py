import jax
import jax.numpy as jnp


class DenseLayer:

    def __init__(self, n, m, edge_kernel, node_kernel):

        self.n = n
        self.m = m

        self.edge_kernel = edge_kernel
        self.node_kernel = node_kernel

        self.edge_parameters = jnp.array(
            [edge_kernel.default_parameters] * (n * m)
        ).reshape(n, m, -1)
        self.node_parameters = jnp.array(
            [node_kernel.default_parameters] * m
        ).reshape(m, -1)

        self.edge_states = jnp.zeros((n, m, 1))
        self.node_states = jnp.zeros((m, 1))

        self.jit_tick = jax.jit(self.tick)

    def __call__(self, input_node_states, use_jit=True):

        tick = self.jit_tick if use_jit else self.tick
        self.edge_states, self.node_states = tick(
            input_node_states,
            self.edge_parameters,
            self.node_parameters)

        return self.node_states

    def tick(self, input_node_states, edge_parameters, node_parameters):

        # input_node_states: (n, k)
        # edge_parameters  : (n, m, r)
        # node_parameters  : (m, s)
        # output           : (m, k)

        #######################
        # compute edge states #
        #######################

        #            node_state x edge_parameters -> edge_state
        # edge_kernel  : (k)_i  x (r)_ij          -> (l)_ij
        # vedge_kernel : (k)_i  x (m, r)_i        -> (m, l)_i
        # vvedge_kernel: (n, k) x (n, m, r)       -> (n, m, l)

        # map over j
        vkernel = jax.vmap(self.edge_kernel, in_axes=(None, 0))
        # map over i
        vvkernel = jax.vmap(vkernel)

        edge_states = vvkernel(input_node_states, edge_parameters)

        #######################
        # compute node states #
        #######################

        #             edge_states x node_parameters -> node_state
        # node_kernel : (n, l)_j  x (s)_j           -> (k)_j
        # vnode_kernel: (n, m, l) x (m, s)          -> (m, k)

        # map over j
        vnode_kernel = jax.vmap(self.node_kernel, in_axes=(1, 0))

        output_node_states = vnode_kernel(edge_states, node_parameters)

        return edge_states, output_node_states