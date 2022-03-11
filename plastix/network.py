class Node:

    def __init__(self):
        pass

    def set_kernel(self, kernel):
        pass


class Edge:

    def __init__(self, u, v):
        self.u = u
        self.v = v

    def set_kernel(self, kernel):
        pass


class Network:

    def __init__(self):
        pass

    def add_nodes(self, num_nodes):
        pass

    def add_edges(self, edges):
        pass

    def tick(self, delta_t=None):
        pass
