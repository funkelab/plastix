class State:
    '''Declaration class for kernel states.

    To be used as class attributes in custom edge and node kernels.

    Args:

        shape (tuple of int):

            The shape of the state.

        init_fun (callable returning a jax tensor):

            An initialization function to populate the inital values for this
            state. Should take ``shape`` as an argument.

        shared (bool):

            Set to if this state is to be shared with other kernels that
            declared it in the same way.
    '''

    def __init__(self, shape, init_fun=None, shared=False):
        self.shape = shape
        self.init_fun = init_fun
        self.shared = shared
