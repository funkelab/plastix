from ..attribute_array_view import AttributeArrayView
from ..kernels.kernel_attribute_view import KernelAttributeView
from .parameters import LayerParameters
from .states import LayerStates
import jax


class SparseLayer:

    def __init__(self, n, m, edge_indices, edge_kernel, node_kernel):
        '''
        edge_indices is jax array of shape (k, 2), where edge_indices[i, 0] is
        the index of the input node and edge_indices[i, 1] is the index of the
        output node of edge i.
        '''

        self.m = m
        self.n = n

        assert len(edge_indices.shape) == 2, \
            "Edge indices should be given as a 2D array of shape (k, 2)."
        assert edge_indices.shape[1] == 2, \
            "Edge indices should be given as a 2D array of shape (k, 2)."

        self.num_edges = edge_indices.shape[0]
        self.u_indices = edge_indices[:, 0]
        self.v_indices = edge_indices[:, 1]
        self.edge_kernel = edge_kernel
        self.node_kernel = node_kernel

        # keep a compiled version of update functions
        self._jit_update_state = jax.jit(self._update_state)
        self._jit_update_parameters = jax.jit(self._update_parameters)

    def init_state(self):

        input_node_states = jax.vmap(
            self.node_kernel.init_state_data,
            axis_size=self.n)()
        output_node_states = jax.vmap(
            self.node_kernel.init_state_data,
            axis_size=self.m)()
        edge_states = jax.vmap(
                self.edge_kernel.init_state_data,
                axis_size=self.num_edges)()

        shared_edge_states = self.edge_kernel.init_shared_state_data()
        shared_node_states = self.node_kernel.init_shared_state_data()

        input_node_state_view = AttributeArrayView(
            self.node_kernel.get_states(shared=False),
            input_node_states)
        shared_node_state_view = AttributeArrayView(
            self.node_kernel.get_states(shared=True),
            shared_node_states)
        output_node_state_view = AttributeArrayView(
            self.node_kernel.get_states(shared=False),
            output_node_states)
        edge_state_view = AttributeArrayView(
            self.edge_kernel.get_states(shared=False),
            edge_states)
        shared_edge_state_view = AttributeArrayView(
            self.edge_kernel.get_states(shared=True),
            shared_edge_states)

        input_nodes = KernelAttributeView([
            input_node_state_view,
            shared_node_state_view])
        output_nodes = KernelAttributeView([
            output_node_state_view,
            shared_node_state_view])
        edges = KernelAttributeView([
            edge_state_view,
            shared_edge_state_view])

        return LayerStates(input_nodes, output_nodes, edges)

    def init_parameters(self):

        output_node_parameters = jax.vmap(
            self.node_kernel.init_parameter_data,
            axis_size=self.m)()
        edge_parameters = jax.vmap(
            self.edge_kernel.init_parameter_data,
            axis_size=self.num_edges)()

        shared_edge_parameters = self.edge_kernel.init_shared_parameter_data()
        shared_node_parameters = self.node_kernel.init_shared_parameter_data()

        output_node_parameter_view = AttributeArrayView(
            self.node_kernel.get_parameters(shared=False),
            output_node_parameters)
        shared_node_parameter_view = AttributeArrayView(
            self.node_kernel.get_parameters(shared=True),
            shared_node_parameters)

        edge_parameter_view = AttributeArrayView(
            self.edge_kernel.get_parameters(shared=False),
            edge_parameters)
        shared_edge_parameter_view = AttributeArrayView(
            self.edge_kernel.get_parameters(shared=True),
            shared_edge_parameters)

        output_nodes = KernelAttributeView([
            output_node_parameter_view,
            shared_node_parameter_view])
        edges = KernelAttributeView([
            edge_parameter_view,
            shared_edge_parameter_view])

        return LayerParameters(output_nodes, edges)

    def update_state(self, state, parameters, use_jit=True):

        if use_jit:
            update_fn = self._jit_update_state
        else:
            update_fn = self._update_state

        return update_fn(state, parameters)

    def update_parameters(self, state, parameters, use_jit=True):

        if use_jit:
            update_fn = self._jit_update_parameters
        else:
            update_fn = self._update_parameters

        return update_fn(state, parameters)

    def _update_state(self, state, parameters):

        input_node_states = state.input_nodes._maps[0]._array

        edge_states = state.edges._maps[0]._array
        shared_edge_states = state.edges._maps[1]._array
        edge_parameters = parameters.edges._maps[0]._array
        shared_edge_parameters = parameters.edges._maps[1]._array

        output_node_states = state.output_nodes._maps[0]._array
        output_node_parameters = parameters.output_nodes._maps[0]._array
        shared_node_states = state.output_nodes._maps[1]._array
        shared_node_parameters = parameters.output_nodes._maps[1]._array

        #######################
        # compute edge states #
        #######################

        # shapes of arrays retrieved above:
        #
        # edge_states           : (k, ?)
        # edge_parameters       : (k, ?)
        # input_node_states     : (n, ?)
        # output_node_states    : (m, ?)
        # output_node_parameters: (m, ?)
        #
        # where k is the number of edges, ? is the size of the state/parameters
        # for this edge/node (all together in one 1D array)

        def edge_kernel(es, ep, ns):
            return self.edge_kernel._update_state(
                self.node_kernel.__class__,
                shared_edge_states,
                shared_edge_parameters,
                es, ep, ns)

        # edge_kernel argument shapes:
        #
        # edge_states       : (?)
        # edge_parameters   : (?)
        # input_node_states : (?)

        # map over all edges i=1,...,k
        vedge_kernel = jax.vmap(edge_kernel)

        # vedge_kernel argument shapes:
        #
        # edge_states       : (k, ?)
        # edge_parameters   : (k, ?)
        # input_node_states : (k, ?)

        edge_states = vedge_kernel(
            edge_states,
            edge_parameters,
            input_node_states[self.u_indices])

        #######################
        # compute node states #
        #######################

        # pass input edge class and shared attributes to node kernel
        def node_kernel(ns, np, es):
            return self.node_kernel._update_state(
                self.edge_kernel.__class__,
                shared_node_states,
                shared_node_parameters,
                ns, np, es)

        # node_kernel argument shapes:
        #
        # node_states    : (?)
        # node_parameters: (?)
        # edge_states    : (n_j, ?)
        #
        # where n_j is the number of incoming edges into output node j

        def update_output_node(ns, np, j):
            return node_kernel(
                ns,
                np,
                edge_states[self.in_nodes[j]])

        # map over j = 1,...,m
        vnode_kernel = jax.vmap(node_kernel)

        # vnode_kernel argument shapes:
        #
        # node_states    : (m, ?)
        # node_parameters: (m, ?)
        # node index     : (m,)

        output_node_states = vnode_kernel(
            output_node_states,
            output_node_parameters,
            self.output_node_indices)

        output_node_state_view = AttributeArrayView(
            self.node_kernel.get_states(shared=False),
            output_node_states)
        shared_node_state_view = AttributeArrayView(
            self.node_kernel.get_states(shared=True),
            shared_node_states)
        edge_state_view = AttributeArrayView(
            self.edge_kernel.get_states(shared=False),
            edge_states)
        shared_edge_state_view = AttributeArrayView(
            self.edge_kernel.get_states(shared=True),
            shared_edge_states)

        output_nodes = KernelAttributeView([
            output_node_state_view,
            shared_node_state_view])
        edges = KernelAttributeView([
            edge_state_view,
            shared_edge_state_view])

        return LayerStates(state.input_nodes, output_nodes, edges)

    def _update_parameters(self, state, parameters):

        input_node_states = state.input_nodes._maps[0]._array

        edge_states = state.edges._maps[0]._array
        shared_edge_states = state.edges._maps[1]._array
        edge_parameters = parameters.edges._maps[0]._array
        shared_edge_parameters = parameters.edges._maps[1]._array

        output_node_states = state.output_nodes._maps[0]._array
        output_node_parameters = parameters.output_nodes._maps[0]._array
        shared_node_states = state.output_nodes._maps[1]._array
        shared_node_parameters = parameters.output_nodes._maps[1]._array

        ###########################
        # compute edge parameters #
        ###########################

        # pass input node class and shared attributes to edge kernel
        def edge_kernel(es, ep, ins, ons):
            return self.edge_kernel._update_parameters(
                self.node_kernel.__class__,
                self.node_kernel.__class__,
                shared_edge_states,
                shared_edge_parameters,
                es, ep, ins, ons)

        # edge_kernel argument shapes:
        #
        # edge_states       : (?)
        # edge_parameters   : (?)
        # input_node_states : (?)
        # output_node_states: (?)

        # map over j = 1,...,m
        vedge_kernel = jax.vmap(edge_kernel, in_axes=(0, 0, None, 0))

        # vedge_kernel argument shapes:
        #
        # edge_states       : (m, ?)
        # edge_parameters   : (m, ?)
        # input_node_states : (?)
        # output_node_states: (m, ?)

        # map over i = 1,...,n
        vvedge_kernel = jax.vmap(vedge_kernel, in_axes=(0, 0, 0, None))

        # vvedge_kernel argument shapes:
        #
        # edge_states       : (n, m, ?)
        # edge_parameters   : (n, m, ?)
        # input_node_states : (n, ?)
        # output_node_states: (m, ?)

        edge_parameters = vvedge_kernel(
            edge_states,
            edge_parameters,
            input_node_states,
            output_node_states)

        #######################
        # compute node states #
        #######################

        # pass input edge class and shared attributes to node kernel
        def node_kernel(ns, np, es):
            return self.node_kernel._update_parameters(
                self.edge_kernel.__class__,
                shared_node_states,
                shared_node_parameters,
                ns, np, es)

        # node_kernel argument shapes:
        #
        # node_states    : (?)
        # node_parameters: (?)
        # edge_states    : (n, ?)

        # map over j = 1,...,m
        vnode_kernel = jax.vmap(node_kernel, in_axes=(0, 0, 1))

        # vnode_kernel argument shapes:
        #
        # node_states    : (m, ?)
        # node_parameters: (m, ?)
        # edge_states    : (n, m, ?)

        output_node_parameters = vnode_kernel(
            output_node_states,
            output_node_parameters,
            edge_states)

        output_node_parameter_view = AttributeArrayView(
            self.node_kernel.get_parameters(shared=False),
            output_node_parameters)
        shared_node_parameter_view = AttributeArrayView(
            self.node_kernel.get_parameters(shared=True),
            shared_node_parameters)

        edge_parameter_view = AttributeArrayView(
            self.edge_kernel.get_parameters(shared=False),
            edge_parameters)
        shared_edge_parameter_view = AttributeArrayView(
            self.edge_kernel.get_parameters(shared=True),
            shared_edge_parameters)

        output_nodes = KernelAttributeView([
            output_node_parameter_view,
            shared_node_parameter_view])
        edges = KernelAttributeView([
            edge_parameter_view,
            shared_edge_parameter_view])

        return LayerParameters(output_nodes, edges)
