from ...attribute_array_view import AttributeArrayView
from ..kernel import Kernel
from ..kernel_attribute_view import KernelAttributeView
import abc


class NodeKernel(Kernel):

    def _update_state(
            self,
            edge_class,
            shared_state_data,
            shared_parameter_data,
            state_data,
            parameter_data,
            edge_states_data):

        self.set_state_data(state_data, shared_state_data)
        self.set_parameter_data(parameter_data, shared_parameter_data)

        edge_state_view = AttributeArrayView(
            edge_class.get_states(shared=False),
            edge_states_data)
        edges = KernelAttributeView([edge_state_view])

        self.update_state(edges)

        state_data = self.get_state_data()

        return state_data

    def _update_parameters(
            self,
            edge_class,
            shared_state_data,
            shared_parameter_data,
            state_data,
            parameter_data,
            edge_states_data):

        self.set_state_data(state_data, shared_state_data)
        self.set_parameter_data(parameter_data, shared_parameter_data)

        edge_state_view = AttributeArrayView(
            edge_class.get_states(shared=False),
            edge_states_data)
        edges = KernelAttributeView([edge_state_view])

        self.update_parameters(edges)

        parameter_data = self.get_parameter_data()

        return parameter_data

    @abc.abstractmethod
    def update_state(self, edges):
        '''Execute this node kernel.

        Args:

            edges (:class:`KernelAttributeView` of edge kernels):

                A thin wrapper providing semantic access to the incoming edge
                states. For each state attribute ``x`` of the incoming edge
                kernel, this wrapper has a tensor ``edges.x``, where the first
                dimension corresponds to the number of incoming edges and the
                remaining dimensions are given by the shape of state ``x``.
        '''
        pass

    @abc.abstractmethod
    def update_parameters(self, edges):
        '''Execute this node kernel.

        Args:

            edges (:class:`KernelAttributeView` of edge kernels):

                A thin wrapper providing semantic access to the incoming edge
                states. For each state attribute ``x`` of the incoming edge
                kernel, this wrapper has a tensor ``edges.x``, where the first
                dimension corresponds to the number of incoming edges and the
                remaining dimensions are given by the shape of state ``x``.
        '''
        pass
