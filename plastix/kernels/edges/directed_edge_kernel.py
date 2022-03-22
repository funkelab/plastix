from .edge_kernel import EdgeKernel
import abc


class DirectedEdgeKernel(EdgeKernel):
    '''Same as :class:`EdgeKernel`, but without access to the outgoing node
    state. More efficient than :class:`EdgeKernel`.
    '''

    def __call__(
            self,
            input_node_class,
            input_node_state_data,
            state_data,
            parameter_data):

        self.set_state_data(state_data)
        self.set_parameter_data(parameter_data)

        input_node = input_node_class()
        input_node.set_state_data(input_node_state_data)

        self.tick(input_node)

        state_data = self.get_state_data()
        parameter_data = self.get_parameter_data()

        return state_data, parameter_data

    @abc.abstractmethod
    def tick(self, input_node):
        '''Execute this edge kernel.

        Args:

            input_node (:class:``NodeKernel``):

                A node kernel, providing semantic access to the input node
                states. For each state attribute ``x`` of the input node
                kernel, this kernel has a tensor ``input_node.x``.
        '''
        pass
