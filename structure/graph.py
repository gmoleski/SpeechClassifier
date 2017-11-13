from collections import defaultdict


class Graph(object):

    def __init__(self, connections, directed=False):
        self.graph = defaultdict(set)
        self.add_connections(connections)
        self.is_directed = directed

    def add_connections(self, connections):
        """ Add connections (list of tuple pairs) to graph """
        for node_1, node_2 in connections:
            self.add(node_1, node_2)

    def add(self, node_1, node_2):
        """ Add connection for node_1 and node_2 """
        self.graph[node_1].add(node_2)
        if not self.graph:
            self.graph[node_2].add(node_1)

    def remove(self, node):
        """ Remove all references to node """
        for n, conn in self.graph.iteritems():
            if node in conn:
                conn.remove(node)
            if node in self.graph:
                del self.graph[node]


    def is_connected(self, node_1, node_2):
        """ Is node_1 directly connected to node_2 """
        return node_1 in self.graph and node_2 in self.graph[node_1]

    def find_path(self, node_1, node_2, path=[]):
        """ Find any path between node_1 and node_2 """

        path = path + [node_1]
        if node_1 == node_2:
            return path
        if node_1 not in self.graph:
            return None
        for node in self.graph[node_1]:
            if node not in path:
                new_path = self.find_path(node, node_2, path)
                if new_path:
                    return new_path
        return None
