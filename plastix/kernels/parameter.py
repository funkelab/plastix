from ..attribute import Attribute


class Parameter(Attribute):
    '''Declaration class for kernel parameters.

    To be used as class attributes in custom edge and node kernels.

    Args:

        shape (tuple of int):

            The shape of the parameter.

        init_fun (callable returning a jax tensor):

            An initialization function to populate the inital values for this
            parameter. Should take ``shape`` as an argument.

        shared (bool):

            Set to if this parameter is to be shared with other kernels that
            declared it in the same way.
    '''

    def __init__(self, shape, init_fun=None, shared=False):

        super().__init__(shape)
        self.init_fun = init_fun
        self.shared = shared

    def __repr__(self):
        return f"{self.shape}, shared={self.shared}"
