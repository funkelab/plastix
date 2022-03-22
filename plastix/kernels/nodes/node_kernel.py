from ..kernel import Kernel
import abc


class NodeKernel(Kernel):

    def __call__(self, edge_states, state_data, parameter_data):

        self.set_state_data(state_data)
        self.set_parameter_data(parameter_data)

        self.tick(edge_states)

        state_data = self.get_state_data()
        parameter_data = self.get_parameter_data()

        return state_data, parameter_data

    @abc.abstractmethod
    def tick(self, edge_states):
        '''Execute this node kernel.

        Args:

            edge_states (tensor of shape ``(n, k)``):

                Tensor of the visible incoming edge states, where ``n`` is the
                number of edges and ``k`` the size of their state vector.
        '''
        pass
