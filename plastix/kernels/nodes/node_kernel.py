from ..kernel import Kernel
import abc


class NodeKernel(Kernel):

    @abc.abstractmethod
    def tick(self, edge_states, node_parameters):
        '''Execute this node kernel.

        Args:

            edge_states (tensor of shape ``(n,k)``):

                Tensor of the visible incoming edge states, where ``n`` is the
                number of edges and ``k`` the size of their state vector.

            node_parameters (tensor of shape ``(l,)``):

                Parameters of this node as a vector of size ``l``.

        Returns:

            A tuple ``(node_state, node_parameters)`` for the updated state and
            parameters of this node.
        '''
        pass
