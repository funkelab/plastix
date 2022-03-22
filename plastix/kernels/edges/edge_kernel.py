from ..kernel import Kernel
import abc


class EdgeKernel(Kernel):

    def __call__(
            self,
            input_node_state,
            output_node_state,
            state_data,
            parameter_data):

        self.set_state_data(state_data)
        self.set_parameter_data(parameter_data)

        self.tick(input_node_state, output_node_state)

        state_data = self.get_state_data()
        parameter_data = self.get_parameter_data()

        return state_data, parameter_data

    @abc.abstractmethod
    def tick(self, input_node_state, output_node_state):
        '''Execute this edge kernel.

        Args:

            input_node_state (tensor of shape ``(k,)``):

                The visible incoming node state as a vector of size ``k``.

            output_node_state (tensor of shape ``(k,)``):

                The visible outgoing node state as a vector of size ``k``.
        '''
        pass
