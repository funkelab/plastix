from ..attribute_array_view import AttributeArrayView
from .parameter import Parameter
from .state import State
import abc
import jax.numpy as jnp


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

    def _set_attribute_data(self, attributes, data):

        attribute_view = AttributeArrayView(attributes, data)

        for name, attribute in attributes.items():
            self.__setattr__(name, attribute_view.__getattr__(name))

        return attribute_view

    def set_state_data(self, data, shared_data=None):

        self.__states = self._set_attribute_data(
            self.get_states(shared=False),
            data)

        if shared_data is not None:
            self.__shared_states = self._set_attribute_data(
                self.get_states(shared=True),
                shared_data)

    def get_state_data(self, include_shared=False):

        # if nothing was changed, no reason to create a new tensor
        if all(
                getattr(self, name) is self.__states.__getattr__(name)
                for name in self.get_states(shared=False).keys()):
            data = self.__states._array

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
            self.get_parameters(shared=False),
            data)

        if shared_data is not None:
            self.__shared_parameters = self._set_attribute_data(
                self.get_parameters(shared=True),
                shared_data)

    def get_parameter_data(self, include_shared=False):

        # if nothing was changed, no reason to create a new tensor
        if all(
                getattr(self, name) is self.__parameters.__getattr__(name)
                for name in self.get_parameters(shared=False).keys()):
            data = self.__parameters._array

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

    def __repr__(self):
        r = f"Kernel of type {self.__class__.__name__} with:"
        r += "\nStates:"
        for name, attribute in self.get_states().items():
            r += f"\n\t{name}: {attribute}"
        r += "\nParameters:"
        for name, attribute in self.get_parameters().items():
            r += f"\n\t{name}: {attribute}"
        r += "\n"
        return r
