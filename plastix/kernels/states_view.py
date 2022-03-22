class StatesView:
    '''Provides a semantic view on a list of kernel states.

    Given the state tensor for a collection of kernels (nodes or edges), this
    view allows accessing individual state tensors by their name as defined in
    the kernel.

    ..code-block:: python

        class ExampleKernel(DirectedEdgeKernel):

            frizz = State((2, 3))
            frotz = State((5,))

            weight = Parameter((1,))

            def tick(self, node):
                self.frizz = node.rate * self.weight

        states_data = jnp.zeros((100, 2, 3))

        view = StatesView(ExampleKernel, states_data)
        view.frizz  # a tensor of shape (100, 2, 3)
        view.frotz  # a tensor of shape (100, 5)
    '''

    def __init__(self, kernel_class, data):

        states = kernel_class.get_states()

        for name, state in states.items():
            state_slice = kernel_class._get_attribute_slice(name)
            state_view = data[:, state_slice].reshape((-1,) + state.shape)
            self.__setattr__(name, state_view)
