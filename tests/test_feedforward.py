import plastix as px
import unittest


class TestFeedforward(unittest.TestCase):

    def __test_creation(self):

        #   0 - 2--
        #    \ /    \
        #     x      4
        #    / \    /
        #   1 - 3--

        network = px.Network()
        nodes = network.add_nodes(5)
        edges = network.add_edges([
            (nodes[0], nodes[2]),
            (nodes[0], nodes[3]),
            (nodes[1], nodes[2]),
            (nodes[1], nodes[3]),
            (nodes[2], nodes[4]),
            (nodes[3], nodes[4])
        ])

        for node in nodes:
            node.set_kernel(px.kernels.nodes.SumNonlinear())
        for edge in edges:
            edge.set_kernel(px.kernels.edges.FixedWeight())

        # initialize kernel parameters
        nodes[0].bias = 0.0
        nodes[1].bias = 0.0
        nodes[2].bias = 0.5
        nodes[3].bias = 1.5
        nodes[4].bias = 0.0
        edges[0].weight = 1.0
        edges[1].weight = 1.0
        edges[2].weight = 1.0
        edges[3].weight = 1.0
        edges[4].weight = 0.5
        edges[5].weight = -0.5

        # clamp input node values
        nodes[0].rate = 0.0
        nodes[1].rate = 0.0

        # update nodes
        network.tick()

        # test output
        assert nodes[4].rate < 0
