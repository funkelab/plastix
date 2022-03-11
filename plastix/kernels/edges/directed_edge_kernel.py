from .edge_kernel import EdgeKernel
import abc


class DirectedEdgeKernel(EdgeKernel):
    '''Same as :class:`EdgeKernel`, but without access to the outgoing node
    state. More efficient than :class:`EdgeKernel`.
    '''

    @abc.abstractmethod
    def tick(self, input_node_state, edge_parameters):
        '''Execute this edge kernel.

        Args:

            input_node_state (tensor of shape ``(k,)``):

                The visible incoming node state as a vector of size ``k``.

            edge_parameters (tensor of shape ``(l,)``):

                Parameters of this edge as a vector of size ``l``.

        Returns:

            A tuple ``(edge_state, edge_parameters)`` for the updated state and
            parameters of this edge.
        '''
        pass
