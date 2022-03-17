from .parameter import Parameter
import abc
import jax.numpy as jnp
import numpy as np


class Kernel(abc.ABC):
    '''Abstract base class for kernels.'''

    @classmethod
    def _parse_parameters(cls):

        # already done earlier?
        if hasattr(cls, '_parameters'):
            return

        # find all Parameter instances of this class
        cls._parameters = {
            name: attribute
            for name, attribute in cls.__dict__.items()
            if isinstance(attribute, Parameter)
        }

        # assign each parameter to a slice in the parameter array
        cls._parameter_slices = {}
        offset = 0
        for name, parameter in cls._parameters.items():
            length = np.prod(parameter.shape)
            cls._parameter_slices[name] = slice(offset, offset + length)

    @classmethod
    def _get_parameter_slice(cls, name):
        cls._parse_parameters()
        return cls._parameter_slices[name]

    @classmethod
    def get_parameters(cls):
        cls._parse_parameters()
        return cls._parameters

    @classmethod
    def init_parameter_data(cls):
        return jnp.concatenate([
            parameter.init_fun(parameter.shape).reshape(-1)
            for _, parameter in cls.get_parameters().items()
        ])

    def set_parameter_data(self, data):

        self.__original_data = data
        self.__original_views = {}
        for name, parameter in self.get_parameters().items():
            parameter_slice = self._get_parameter_slice(name)
            parameter_view = data[parameter_slice].reshape(parameter.shape)
            self.__setattr__(name, parameter_view)
            self.__original_views[name] = parameter_view

    def get_parameter_data(self):

        # if nothing was changed, no reason to create a new tensor
        if all(
                getattr(self, name) is self.__original_views[name]
                for name in self.get_parameters().keys()):
            return self.__original_data

        return jnp.concatenate([
            getattr(self, name).reshape(-1)
            for name, _ in self.get_parameters().items()
        ])

    def __getattr__(self, name):

        if name in self.get_parameters():

            raise AttributeError(
                f"Asked for parameter '{name}', but parameter data has not "
                "been set yet. Call set_parameter_data() before accessing "
                "parameters.")

        raise AttributeError()
