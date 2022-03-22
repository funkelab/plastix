from ..kernel import Kernel
from ..states_view import StatesView
import abc


class NodeKernel(Kernel):

    def __call__(
            self,
            edge_class,
            edge_states_data,
            state_data,
            parameter_data):

        self.set_state_data(state_data)
        self.set_parameter_data(parameter_data)

        edges = StatesView(edge_class, edge_states_data)

        self.tick(edges)

        state_data = self.get_state_data()
        parameter_data = self.get_parameter_data()

        return state_data, parameter_data

    @abc.abstractmethod
    def tick(self, edges):
        '''Execute this node kernel.

        Args:

            edges (:class:`StatesView` of edge kernels):

                A thin wrapper providing semantic access to the incoming edge
                states. For each state attribute ``x`` of the incoming edge
                kernel, this wrapper has a tensor ``edges.x``, where the first
                dimension corresponds to the number of incoming edges and the
                remaining dimensions are given by the shape of state ``x``.
        '''
        pass
