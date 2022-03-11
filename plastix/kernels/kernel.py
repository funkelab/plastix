import abc


class Kernel(abc.ABC):
    '''Abstract base class for kernels.'''

    def __call__(self, *args, **kwargs):
        '''Convenience method to call the kernel. Forwards calls to tick().'''
        return self.tick(*args, **kwargs)

    @abc.abstractmethod
    def tick(self, *args, **kwargs):
        '''Execute this kernel.'''
        pass
