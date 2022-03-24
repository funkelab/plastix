from .parameter import Parameter
from .state import State
import abc
import jax.numpy as jnp
import numpy as np


class KernelAttributes:

    def __init__(self):
        self.data = None
        self.shared_data = None
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
        shared_offset = 0
        for name, state in cls._states.items():
            length = np.prod(state.shape)
            if state.shared:
                begin = shared_offset
                end = shared_offset + length
                shared_offset += length
            else:
                begin = offset
                end = offset + length
                offset += length
            cls._attribute_slices[name] = slice(begin, end)

        # assign each parameter to a slice in the parameter array
        offset = 0
        shared_offset = 0
        for name, parameter in cls._parameters.items():
            length = np.prod(parameter.shape)
            if state.shared:
                begin = shared_offset
                end = shared_offset + length
                shared_offset += length
            else:
                begin = offset
                end = offset + length
                offset += length
            cls._attribute_slices[name] = slice(begin, end)

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
    def _get_attributes(cls, attributes, shared=None):

        if shared is None:
            return attributes

        return {
            name: attribute
            for name, attribute in attributes.items()
            if attribute.shared == shared
        }

    @classmethod
    def get_states(cls, shared=None):
        '''Get a list of state attributes of this kernel.

        Args:

            shared (bool, default ``None``):

                If ``True``, return only shared states. If ``False``, return
                only non-shared state. If ``None`` (default), return both.
        '''
        cls._parse_attributes()
        return cls._get_attributes(cls._states, shared)

    @classmethod
    def init_state_data(cls):
        return cls._init_attribute_data(cls.get_states(shared=False).values())

    @classmethod
    def init_shared_state_data(cls):
        return cls._init_attribute_data(cls.get_states(shared=True).values())

    def _set_attribute_data(self, attributes, data, shared_data):

        kernel_attributes = KernelAttributes()
        kernel_attributes.data = data
        kernel_attributes.shared_data = shared_data
        kernel_attributes.views = {}

        for name, attribute in attributes.items():
            attribute_slice = self._get_attribute_slice(name)
            if attribute.shared:
                attribute_view = \
                    shared_data[attribute_slice].reshape(attribute.shape)
            else:
                attribute_view = data[attribute_slice].reshape(attribute.shape)
            self.__setattr__(name, attribute_view)
            kernel_attributes.views[name] = attribute_view

        return kernel_attributes

    def set_state_data(self, data, shared_data=None):
        self.__states = self._set_attribute_data(
            self.get_states(),
            data,
            shared_data)

    def get_state_data(self, include_shared=False):

        # if nothing was changed, no reason to create a new tensor
        if all(
                getattr(self, name) is self.__states.views[name]
                for name in self.get_states(shared=False).keys()):
            data = self.__states.data

        else:

            data = jnp.concatenate([
                getattr(self, name).reshape(-1)
                for name, _ in self.get_states(shared=False).items()
            ])

        if not include_shared:
            return data

        shared_data = jnp.concatenate([
            getattr(self, name).reshape(-1)
            for name, _ in self.get_states(shared=True).items()
        ])

        return data, shared_data

    @classmethod
    def get_parameters(cls, shared=None):
        '''Get a list of parameter attributes of this kernel.

        Args:

            shared (bool, default ``None``):

                If ``True``, return only shared parameters. If ``False``,
                return only non-shared parameter. If ``None`` (default), return
                both.
        '''
        cls._parse_attributes()
        return cls._get_attributes(cls._parameters, shared)

    @classmethod
    def init_parameter_data(cls):
        return cls._init_attribute_data(
            cls.get_parameters(shared=False).values()
        )

    @classmethod
    def init_shared_parameter_data(cls):
        return cls._init_attribute_data(
            cls.get_parameters(shared=True).values()
        )

    def set_parameter_data(self, data, shared_data=None):
        self.__parameters = self._set_attribute_data(
            self.get_parameters(),
            data,
            shared_data)

    def get_parameter_data(self, include_shared=False):

        # if nothing was changed, no reason to create a new tensor
        if all(
                getattr(self, name) is self.__parameters.views[name]
                for name in self.get_parameters(shared=False).keys()):
            data = self.__parameters.data

        else:

            data = jnp.concatenate([
                getattr(self, name).reshape(-1)
                for name, _ in self.get_parameters(shared=False).items()
            ])

        if not include_shared:
            return data

        shared_data = jnp.concatenate([
            getattr(self, name).reshape(-1)
            for name, _ in self.get_parameters(shared=True).items()
        ])

        return data, shared_data

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
