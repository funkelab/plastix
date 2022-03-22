import jax


class DenseLayer:

    def __init__(self, n, m, edge_kernel, node_kernel):

        self.n = n
        self.m = m

        self.edge_kernel = edge_kernel
        self.node_kernel = node_kernel

        # create state and parameter tensors for all kernels:
        self.edge_states = jax.vmap(
            jax.vmap(
                edge_kernel.init_state_data,
                axis_size=m),
            axis_size=n)()
        self.node_states = jax.vmap(
            node_kernel.init_state_data,
            axis_size=m)()
        self.edge_parameters = jax.vmap(
            jax.vmap(
                edge_kernel.init_parameter_data,
                axis_size=m),
            axis_size=n)()
        self.node_parameters = jax.vmap(
            node_kernel.init_parameter_data,
            axis_size=m)()

        # keep a compiled version of tick()
        self.jit_tick = jax.jit(self.tick)

    def __call__(self, input_node_states, use_jit=True):

        tick = self.jit_tick if use_jit else self.tick

        edge_update, node_update = tick(
            input_node_states,
            self.edge_states,
            self.edge_parameters,
            self.node_states,
            self.node_parameters)

        self.edge_states, self.edge_parameters = edge_update
        self.node_states, self.node_parameters = node_update

        return self.node_states

    def tick(
            self,
            input_node_states,
            edge_states,
            edge_parameters,
            node_states,
            node_parameters):

        # input_node_states: (n, k)
        # edge_states      : (n, m, q)
        # edge_parameters  : (n, m, r)
        # node_states      : (m, k)
        # node_parameters  : (m, s)
        # output           : (m, k)

        #######################
        # compute edge states #
        #######################

        # pass input node class to edge kernel
        def edge_kernel(ns, es, ep):
            return self.edge_kernel(
                self.node_kernel.__class__,
                ns, es, ep)

        # node_states, edge_{states,parameters} -> edge_{states,parameters}
        # edge_kernel  : (k)_i,  (q)_ij,    (r)_ij    -> (q)_ij, (r)_ij
        # vedge_kernel : (k)_i,  (m, q)_i,  (m, r)_i  -> (m, q)_i, (m, r)_i
        # vvedge_kernel: (n, k), (n, m, q), (n, m, r) -> (n, m, q), (n, m, r)

        # map over j
        vkernel = jax.vmap(edge_kernel, in_axes=(None, 0, 0))
        # map over i
        vvkernel = jax.vmap(vkernel)

        edge_states, edge_parameters = vvkernel(
            input_node_states,
            edge_states,
            edge_parameters)
        edge_update = (edge_states, edge_parameters)

        #######################
        # compute node states #
        #######################

        # pass input edge class to node kernel
        def node_kernel(es, ns, np):
            return self.node_kernel(
                self.edge_kernel.__class__,
                es, ns, np)

        # edge_states, node_{states,parameters} -> node_{states,parameters}
        # node_kernel : (n, l)_j,  (k)_j,  (s)_j  -> (k)_j,  (s)_j
        # vnode_kernel: (n, m, l), (m, k), (m, s) -> (m, k), (m, s)

        # map over j
        vnode_kernel = jax.vmap(node_kernel, in_axes=(1, 0, 0))

        node_states, node_parameters = vnode_kernel(
            edge_states,
            node_states,
            node_parameters)
        node_udpate = (node_states, node_parameters)

        return edge_update, node_udpate
