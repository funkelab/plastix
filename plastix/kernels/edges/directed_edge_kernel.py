from .edge_kernel import EdgeKernel
import abc


class DirectedEdgeKernel(EdgeKernel):
    '''Same as :class:`EdgeKernel`, but without access to the outgoing node
    state. More efficient than :class:`EdgeKernel`.
    '''

    def __call__(
            self,
            input_node_state,
            parameter_data):

        self.set_parameter_data(parameter_data)
        output = self.tick(input_node_state)
        parameter_data = self.get_parameter_data()

        return output, parameter_data

    @abc.abstractmethod
    def tick(self, input_node_state):
        '''Execute this edge kernel.

        Args:

            input_node_state (tensor of shape ``(k,)``):

                The visible incoming node state as a vector of size ``k``.

        Returns:

            The updated state of this edge.
        '''
        pass
