from .parameter import Parameter
from .state import State
import abc
import jax.numpy as jnp
import numpy as np


class KernelAttributes:

    def __init__(self):
        self.data = None
        self.views = None


class Kernel(abc.ABC):
    '''Abstract base class for kernels.'''

    @classmethod
    def _parse_attributes(cls):

        # already done earlier?
        if hasattr(cls, '_parameters'):
            return

        # find all State instances of this class
        cls._states = {
            name: attribute
            for name, attribute in cls.__dict__.items()
            if isinstance(attribute, State)
        }

        # find all Parameter instances of this class
        cls._parameters = {
            name: attribute
            for name, attribute in cls.__dict__.items()
            if isinstance(attribute, Parameter)
        }

        # assign each state to a slice in the state array
        cls._attribute_slices = {}
        offset = 0
        for name, state in cls._states.items():
            length = np.prod(state.shape)
            cls._attribute_slices[name] = slice(offset, offset + length)

        # assign each parameter to a slice in the parameter array
        offset = 0
        for name, parameter in cls._parameters.items():
            length = np.prod(parameter.shape)
            cls._attribute_slices[name] = slice(offset, offset + length)

    @classmethod
    def _get_attribute_slice(cls, name):
        cls._parse_attributes()
        return cls._attribute_slices[name]

    @classmethod
    def _init_attribute_data(cls, attributes):

        if not attributes:
            return jnp.zeros((0,))

        return jnp.concatenate([
            attribute.init_fun(attribute.shape).reshape(-1)
            for attribute in attributes
        ])

    @classmethod
    def get_states(cls):
        cls._parse_attributes()
        return cls._states

    @classmethod
    def init_state_data(cls):
        return cls._init_attribute_data(cls.get_states().values())

    def _set_attribute_data(self, attributes, data):

        kernel_attributes = KernelAttributes()
        kernel_attributes.data = data
        kernel_attributes.views = {}

        for name, attribute in attributes.items():
            attribute_slice = self._get_attribute_slice(name)
            attribute_view = data[attribute_slice].reshape(attribute.shape)
            self.__setattr__(name, attribute_view)
            kernel_attributes.views[name] = attribute_view

        return kernel_attributes

    def set_state_data(self, data):
        self.__states = self._set_attribute_data(
            self.get_states(),
            data)

    def get_state_data(self):

        # if nothing was changed, no reason to create a new tensor
        if all(
                getattr(self, name) is self.__states.views[name]
                for name in self.get_states().keys()):
            return self.__states.data

        return jnp.concatenate([
            getattr(self, name).reshape(-1)
            for name, _ in self.get_states().items()
        ])

    @classmethod
    def get_parameters(cls):
        cls._parse_attributes()
        return cls._parameters

    @classmethod
    def init_parameter_data(cls):
        return cls._init_attribute_data(cls.get_parameters().values())

    def set_parameter_data(self, data):
        self.__parameters = self._set_attribute_data(
            self.get_parameters(),
            data)

    def get_parameter_data(self):

        # if nothing was changed, no reason to create a new tensor
        if all(
                getattr(self, name) is self.__parameters.views[name]
                for name in self.get_parameters().keys()):
            return self.__parameters.data

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

        elif name in self.get_states():

            raise AttributeError(
                f"Asked for state '{name}', but state data has not "
                "been set yet. Call set_state_data() before accessing "
                "states.")

        raise AttributeError(
                f"Object of type {self.__class__} has no attribute {name}")
