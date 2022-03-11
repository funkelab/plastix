from ..kernel import Kernel
import abc


class EdgeKernel(Kernel):

    @abc.abstractmethod
    def tick(self, input_node_state, output_node_state, edge_parameters):
        '''Execute this edge kernel.

        Args:

            input_node_state (tensor of shape ``(k,)``):

                The visible incoming node state as a vector of size ``k``.

            output_node_state (tensor of shape ``(k,)``):

                The visible outgoing node state as a vector of size ``k``.

            edge_parameters (tensor of shape ``(l,)``):

                Parameters of this edge as a vector of size ``l``.

        Returns:

            A tuple ``(edge_state, edge_parameters)`` for the updated state and
            parameters of this edge.
        '''
        pass
