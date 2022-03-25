from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class LayerStates:

    def __init__(self, input_nodes, output_nodes, edges):
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.edges = edges

    def tree_flatten(self):
        children = (
            self.input_nodes,
            self.output_nodes,
            self.edges
        )
        return (children, None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def __repr__(self):
        r = f"input_nodes = ({self.input_nodes}), "
        r += f"output_nodes = ({self.output_nodes}), "
        r += f"edges = ({self.edges})"
        return r
