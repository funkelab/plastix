class Attribute:
    '''Declaration class for attributes.

    Args:

        shape (tuple of int):

            The shape of the state.
    '''

    def __init__(self, shape):
        self.shape = shape

    def __repr__(self):
        return f"{self.shape}"
