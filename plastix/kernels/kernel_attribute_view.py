from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class KernelAttributeView:

    def __init__(
            self,
            attribute_array_maps):
        self._maps = attribute_array_maps

    def __getattr__(self, name):

        for attribute_map in self._maps:
            if hasattr(attribute_map, name):
                return attribute_map.__getattr__(name)

        return super().__getattribute__(name)

    def __setattr__(self, name, value):

        if name == '_maps':
            return super().__setattr__(name, value)

        for attribute_map in self._maps:
            if hasattr(attribute_map, name):
                return attribute_map.__setattr__(name, value)

        raise RuntimeError(
            f"Trying to set attribute {name}, but this attribute does not "
            "exist.")

    def tree_flatten(self):

        children = self._maps
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        obj._maps = children
        return obj
