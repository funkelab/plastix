class Parameter:
    '''Declaration class for kernel parameters.

    To be used as class attributes in custom edge and node kernels.

    Args:

        shape (tuple of int):

            The shape of the parameter.

        init_fun (callable returning a jax tensor):

            An initialization function to populate the inital values for this
            parameter.
    '''

    def __init__(self, shape, init_fun=None):
        self.shape = shape
        self.init_fun = init_fun
