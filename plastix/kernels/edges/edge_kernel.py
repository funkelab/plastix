from ..kernel import Kernel
import abc


class EdgeKernel(Kernel):
    '''Same as :class:`EdgeKernel`, but without access to the outgoing node
    state. More efficient than :class:`EdgeKernel`.
    '''

    def _update_state(
            self,
            input_node_class,
            shared_state_data,
            shared_parameter_data,
            state_data,
            parameter_data,
            input_node_state_data):

        self.set_state_data(state_data, shared_state_data)
        self.set_parameter_data(parameter_data, shared_parameter_data)

        input_node = input_node_class()
        input_node.set_state_data(input_node_state_data)

        self.update_state(input_node)

        state_data = self.get_state_data()

        return state_data

    def _update_parameters(
            self,
            input_node_class,
            output_node_class,
            shared_state_data,
            shared_parameter_data,
            state_data,
            parameter_data,
            input_node_state_data,
            output_node_state_data):

        self.set_state_data(state_data, shared_state_data)
        self.set_parameter_data(parameter_data, shared_parameter_data)

        input_node = input_node_class()
        input_node.set_state_data(input_node_state_data)
        output_node = output_node_class()
        output_node.set_state_data(output_node_state_data)

        self.update_parameters(input_node, output_node)

        parameter_data = self.get_parameter_data()

        return parameter_data

    @abc.abstractmethod
    def update_state(self, input_node):
        '''Execute this edge kernel.

        Args:

            input_node (:class:``NodeKernel``):

                A node kernel, providing semantic access to the input node
                states. For each state attribute ``x`` of the input node
                kernel, this kernel has a tensor ``input_node.x``.
        '''
        pass

    @abc.abstractmethod
    def update_parameters(self, input_node, output_node):
        '''Execute this edge kernel.

        Args:

            input_node (:class:``NodeKernel``):

                A node kernel, providing semantic access to the input node
                states. For each state attribute ``x`` of the input node
                kernel, this kernel has a tensor ``input_node.x``.

            output_node (:class:``NodeKernel``):

                A node kernel, providing semantic access to the output node
                states. For each state attribute ``x`` of the output node
                kernel, this kernel has a tensor ``output_node.x``.
        '''
        pass
