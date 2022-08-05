from jax.tree_util import register_pytree_node_class
import numpy as np


@register_pytree_node_class
class AttributeArrayView:
    """Mapping from named attributes to slices in an array.

    Allows accessing (reading and writing) of attribute values through their
    names.

    Args:

        attributes (``dict`` of ``str`` to :class:`Attribute`):

            The attributes to map to the array. The size of all attributes (sum
            of the products of their shape) should equal the size of the last
            dimension of ``array``.

        array (array-like):

            The array to map the attributes to. The shape should be ``(s_1,
            ..., s_n, l)``, where ``l`` is the size of all attributes.
    """

    def __init__(self, attributes, array):

        if len(array.shape) == 1:
            self._batch_slices = None
        else:
            self._batch_slices = tuple(slice(None) for _ in array.shape[:-1])

        total_size = sum(
            np.prod(attribute.shape) for attribute in attributes.values()
        )
        if total_size != array.shape[-1]:
            raise RuntimeError(
                f"Attribute array has a size of {array.shape}, but attributes "
                f"have a total size of {total_size}. The last dimension of "
                "the attribute array has to match the total size."
            )

        self._array = array
        self._slices = {}
        self._shapes = {}
        offset = 0
        for name, attribute in attributes.items():
            size = np.prod(attribute.shape)
            begin = offset
            end = offset + size
            offset += size
            if self._batch_slices is None:
                self._slices[name] = slice(begin, end)
                self._shapes[name] = attribute.shape
            else:
                self._slices[name] = self._batch_slices + (slice(begin, end),)
                self._shapes[name] = array.shape[:-1] + attribute.shape

    def __getattr__(self, name):

        if name not in self._slices:
            return super().__getattribute__(name)

        shape = self._shapes[name]
        slices = self._slices[name]

        return self._array[slices].reshape(shape)

    def __setattr__(self, name, value):

        if (
            name in ["_batch_slices", "_array", "_slices", "_shapes"]
            or name not in self._slices
        ):
            return super().__setattr__(name, value)

        shape = self._shapes[name]
        slices = self._slices[name]

        if self._batch_slices is None:
            value = value.reshape(-1)
        else:
            batch_dims = len(self._batch_slices)
            # value = (n, m, k_1, ..., k_n)
            value = value.reshape(shape[:batch_dims] + (-1,))
            # value = (n, m, k)

        self._array = self._array.at[slices].set(value)

    def tree_flatten(self):
        return (
            (self._array,),
            (self._slices, self._shapes, self._batch_slices),
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        obj._array = children[0]
        obj._slices, obj._shapes, obj._batch_slices = aux_data
        return obj

    def __repr__(self):
        return f"{self.__dict__}"
