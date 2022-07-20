from ..attribute_array_view import AttributeArrayView
from ..kernels.kernel_attribute_view import KernelAttributeView
from .parameters import LayerParameters
from .states import LayerStates
import jax
import jax.numpy as jnp


class SparseLayer:
    def __init__(self, n, m, edges, edge_kernel, node_kernel):
        """Create a sparse layer.

        Args:

            n (int):
                Number of input nodes.

            m (int):
                Number of output nodes.

            edges (list of tuples):
                List of edges. Each edge is a tuple, containing the index of an
                input node and an output node.

            edge_kernel (:class:`EdgeKernel`):
                The edge kernel to use for all edges.

            node_kernel (:class:`NodeKernel`):
                The node kernel to use for the output nodes.
        """

        self.m = m
        self.n = n
        assert len(edges) > 0, "Number of edges in the graph cannot be zero"
        # sort the edges by output node so as to have continuous allocation
        self.edges = sorted(edges, key=lambda x: x[1])
        self.num_edges = len(edges)
        self.edge_kernel = edge_kernel
        self.node_kernel = node_kernel

        self.u_indices = jnp.array(list(u for u, _ in self.edges))
        self.v_indices = jnp.array(list(v for _, v in self.edges))

        max_u = jnp.max(self.u_indices)
        max_v = jnp.max(self.v_indices)

        assert (
            max_u <= self.n
        ), f"Invalid input node index {max_u} in edge list ({max_u}>{self.n})"
        assert max_v <= self.m, (
            f"Invalid output node index {max_v} in edge list "
            "({max_v}>{self.m})"
        )

        # index range of incoming edges for each output node
        edge_index_range = []
        _start_pointer = _end_pointer = 0

        for j in range(m):
            while self.edges[_end_pointer][1] == j:
                _end_pointer += 1
                if _end_pointer == len(self.edges):
                    break
            edge_index_range.append((_start_pointer, _end_pointer))
            _start_pointer = _end_pointer

        # convert to tuple of tuples to be hashable
        self.edge_index_range = tuple(edge_index_range)
        assert len(self.edge_index_range) == self.m, (
            f"Length of edge index range should be equal to the number of \
            output nodes {len(self.edge_index_range)}"
            "({len(self.edge_index_range)}>{self.m})"
        )

        # keep a compiled version of update functions
        self._jit_update_state = jax.jit(self._update_state)
        self._jit_update_parameters = jax.jit(self._update_parameters)

    def init_state(self):

        input_node_states = jax.vmap(
            self.node_kernel.init_state_data, axis_size=self.n
        )()
        output_node_states = jax.vmap(
            self.node_kernel.init_state_data, axis_size=self.m
        )()
        edge_states = jax.vmap(
            self.edge_kernel.init_state_data, axis_size=self.num_edges
        )()

        shared_edge_states = self.edge_kernel.init_shared_state_data()
        shared_node_states = self.node_kernel.init_shared_state_data()

        input_node_state_view = AttributeArrayView(
            self.node_kernel.get_states(shared=False), input_node_states
        )
        shared_node_state_view = AttributeArrayView(
            self.node_kernel.get_states(shared=True), shared_node_states
        )
        output_node_state_view = AttributeArrayView(
            self.node_kernel.get_states(shared=False), output_node_states
        )
        edge_state_view = AttributeArrayView(
            self.edge_kernel.get_states(shared=False), edge_states
        )
        shared_edge_state_view = AttributeArrayView(
            self.edge_kernel.get_states(shared=True), shared_edge_states
        )

        input_nodes = KernelAttributeView(
            [input_node_state_view, shared_node_state_view]
        )
        output_nodes = KernelAttributeView(
            [output_node_state_view, shared_node_state_view]
        )
        edges = KernelAttributeView([edge_state_view, shared_edge_state_view])

        return LayerStates(input_nodes, output_nodes, edges)

    def init_parameters(self):

        output_node_parameters = jax.vmap(
            self.node_kernel.init_parameter_data, axis_size=self.m
        )()
        edge_parameters = jax.vmap(
            self.edge_kernel.init_parameter_data, axis_size=self.num_edges
        )()

        shared_edge_parameters = self.edge_kernel.init_shared_parameter_data()
        shared_node_parameters = self.node_kernel.init_shared_parameter_data()

        output_node_parameter_view = AttributeArrayView(
            self.node_kernel.get_parameters(shared=False),
            output_node_parameters,
        )
        shared_node_parameter_view = AttributeArrayView(
            self.node_kernel.get_parameters(shared=True),
            shared_node_parameters,
        )

        edge_parameter_view = AttributeArrayView(
            self.edge_kernel.get_parameters(shared=False), edge_parameters
        )
        shared_edge_parameter_view = AttributeArrayView(
            self.edge_kernel.get_parameters(shared=True),
            shared_edge_parameters,
        )

        output_nodes = KernelAttributeView(
            [output_node_parameter_view, shared_node_parameter_view]
        )
        edges = KernelAttributeView(
            [edge_parameter_view, shared_edge_parameter_view]
        )

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
                es,
                ep,
                ns,
            )

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
            edge_states, edge_parameters, input_node_states[self.u_indices]
        )

        #######################
        # compute node states #
        #######################

        # pass input edge class and shared attributes to node kernel

        # node_kernel argument shapes:
        #
        # node_states    : (?)
        # node_parameters: (?)
        # edge_states    : (n_j, ?)
        #
        # where n_j is the number of incoming edges into output node j

        def node_kernel(ns, np, es):
            return self.node_kernel._update_state(
                self.edge_kernel.__class__,
                shared_node_states,
                shared_node_parameters,
                ns,
                np,
                es,
            )

        def update_output_nodes(
            output_node_states,
            output_node_parameters,
            edge_states,
            edge_index_range,
        ):
            return jnp.array(
                [
                    node_kernel(
                        output_node_states[j],
                        output_node_parameters[j],
                        edge_states[
                            edge_index_range[j][0]: edge_index_range[j][1]
                        ],
                    )
                    for j in range(self.m)
                ]
            )

        jit_update_output_nodes = jax.jit(
            update_output_nodes, static_argnames=("edge_index_range",)
        )

        output_node_states = jit_update_output_nodes(
            output_node_states,
            output_node_parameters,
            edge_states,
            self.edge_index_range,
        )

        output_node_state_view = AttributeArrayView(
            self.node_kernel.get_states(shared=False), output_node_states
        )
        shared_node_state_view = AttributeArrayView(
            self.node_kernel.get_states(shared=True), shared_node_states
        )
        edge_state_view = AttributeArrayView(
            self.edge_kernel.get_states(shared=False), edge_states
        )
        shared_edge_state_view = AttributeArrayView(
            self.edge_kernel.get_states(shared=True), shared_edge_states
        )

        output_nodes = KernelAttributeView(
            [output_node_state_view, shared_node_state_view]
        )
        edges = KernelAttributeView([edge_state_view, shared_edge_state_view])

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

        # pass input, output node class and shared attributes to edge kernel
        def edge_kernel(es, ep, ins, ons):
            return self.edge_kernel._update_parameters(
                self.node_kernel.__class__,
                self.node_kernel.__class__,
                shared_edge_states,
                shared_edge_parameters,
                es,
                ep,
                ins,
                ons,
            )

        # edge_kernel argument shapes:
        #
        # edge_states       : (?)
        # edge_parameters   : (?)
        # input_node_states : (?)
        # output_node_states: (?)

        # map over j = 1,...k (number of edges)
        vedge_kernel = jax.vmap(edge_kernel, in_axes=(0, 0, 0, 0))

        # vedge_kernel argument shapes:
        #
        # edge_states       : (k, ?)
        # edge_parameters   : (k, ?)
        # input_node_states : (k, ?)
        # output_node_states: (k, ?)

        edge_parameters = vedge_kernel(
            edge_states,
            edge_parameters,
            input_node_states[self.u_indices],
            output_node_states[self.v_indices],
        )

        #######################
        # compute node states #
        #######################

        # pass input edge class and shared attributes to node kernel
        def node_kernel(ns, np, es):
            return self.node_kernel._update_parameters(
                self.edge_kernel.__class__,
                shared_node_states,
                shared_node_parameters,
                ns,
                np,
                es,
            )

        # node_kernel argument shapes:
        #
        # node_states    : (?)
        # node_parameters: (?)
        # edge_states    : (n_j, ?)
        # where n_j is the number of incoming edges into output node j

        def update_output_nodes(
            output_node_states,
            output_node_parameters,
            edge_states,
            edge_index_range,
        ):
            return jnp.array(
                [
                    node_kernel(
                        output_node_states[j],
                        output_node_parameters[j],
                        edge_states[
                            edge_index_range[j][0]: edge_index_range[j][1]
                        ],
                    )
                    for j in range(self.m)
                ]
            )

        jit_update_output_nodes = jax.jit(
            update_output_nodes, static_argnames=("edge_index_range",)
        )

        output_node_parameters = jit_update_output_nodes(
            output_node_states,
            output_node_parameters,
            edge_states,
            self.edge_index_range,
        )

        output_node_parameter_view = AttributeArrayView(
            self.node_kernel.get_parameters(shared=False),
            output_node_parameters,
        )
        shared_node_parameter_view = AttributeArrayView(
            self.node_kernel.get_parameters(shared=True),
            shared_node_parameters,
        )

        edge_parameter_view = AttributeArrayView(
            self.edge_kernel.get_parameters(shared=False), edge_parameters
        )
        shared_edge_parameter_view = AttributeArrayView(
            self.edge_kernel.get_parameters(shared=True),
            shared_edge_parameters,
        )

        output_nodes = KernelAttributeView(
            [output_node_parameter_view, shared_node_parameter_view]
        )
        edges = KernelAttributeView(
            [edge_parameter_view, shared_edge_parameter_view]
        )

        return LayerParameters(output_nodes, edges)
