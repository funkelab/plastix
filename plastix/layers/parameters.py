from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class LayerParameters:

    def __init__(self, output_nodes, edges):
        self.output_nodes = output_nodes
        self.edges = edges

    def tree_flatten(self):
        children = (
            self.output_nodes,
            self.edges
        )
        return (children, None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def __repr__(self):
        r = f"output_nodes = ({self.output_nodes}), "
        r += f"edges = ({self.edges})"
        return r
