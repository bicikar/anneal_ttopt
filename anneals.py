'''
Stores graph vertices and edges with cost
'''
class Graph:
    def __init__(self, num_of_nodes):
        self.num_of_nodes = num_of_nodes
        self.adj_list = {node: {(node, 0)} for node in range(num_of_nodes)}
        
    def add_edge(self, node1, node2, weight=1):
        self.adj_list[node1].add((node2, weight))
        self.adj_list[node2].add((node1, weight))
        
    def connect_cost(self, node1, node2):
        cost = -1
        for node, n_cost in self.adj_list[node1]:
            if node2 == node:
                cost = n_cost
        return cost
    
    
class HopBased:
    def __init__(self, P, graph, start=None, end=None):
        self.P = P
        self.graph = graph
        self.num_of_nodes = graph.num_of_nodes
        if start is None:
            self.start = 0
        else:
            self.start = start
        if end is None:
            self.end = graph.num_of_nodes - 1
        else:
            self.end = end
        self.Q = [[0 for row in range(P * graph.num_of_nodes)]
                 for col in range(P * graph.num_of_nodes)]
        self.alpha = self.count_alpha()
        
    def get_index(self, i, p):
        return i * self.P + p
    
    def count_alpha(self):
        sum = 0
        for key, val in self.graph.adj_list.items():
            for vert, price in val:
                sum += price
        return sum // 2 + 1
    
    def calc_h_p(self):
        for p in range(self.P):
            for el1 in range(self.num_of_nodes):
                for el2 in range(self.num_of_nodes):
                    sign = -1 if el1 == el2 else 1
                    self.Q[self.get_index(el1, p)][self.get_index(el2, p)] += sign * self.alpha
    
    def calc_h_c(self):
        for p in range(self.P - 1):
            for x1 in range(self.num_of_nodes):
                for x2 in range(self.num_of_nodes):
                    conn_cost = self.graph.connect_cost(x1, x2)
                    cost = conn_cost if conn_cost > -1 else self.alpha
                    self.Q[self.get_index(x1, p)][self.get_index(x2, p + 1)] += cost
                    self.Q[self.get_index(x2, p + 1)][self.get_index(x1, p)] += cost
    def calc_h_s_t(self):
        start = self.get_index(self.start, 0)
        end = self.get_index(self.end, self.P - 1)
        self.Q[start][start] -= self.alpha
        self.Q[end][end] -= self.alpha
        

class EdgeBased:
    def __init__(self, graph, start=None, end=None):
        self.graph = graph
        self.num_of_nodes = graph.num_of_nodes
        self.num_of_edges = ((sum(len(val) for el, val in graph.adj_list.items())) - len(graph.adj_list)) // 2
        self.edges = self.calculate_edges()
        if start is None:
            self.start = 0
        else:
            self.start = start
        if end is None:
            self.end = graph.num_of_nodes - 1
        else:
            self.end = end
        self.Q = [[0 for row in range(graph.num_of_nodes + self.num_of_edges)]
                 for col in range(graph.num_of_nodes + self.num_of_edges)]
        self.max_cost = self.max_cost()
        self.alpha = self.count_alpha()
        
        
    def calculate_edges(self):
        edges = []
        for v in range(self.num_of_nodes):
            for vert, cost in self.graph.adj_list[v]:
                if (vert, v) not in edges and vert != v:
                    edges.append((v, vert))
        return edges
    
    def max_cost(self):
        max_cost = 0
        for key, edges in self.graph.adj_list.items():
            for _, cost in edges:
                max_cost = max(max_cost, cost)
        return max_cost
    
    
    def count_alpha(self):
        sum = 0.
        for key, val in self.graph.adj_list.items():
            for vert, price in val:
                sum += price
        return sum // 2 + 1
    
    def edge_index(self, edge):
        if edge[0] > edge[1]:
            edge = tuple(reversed(edge))
        index = self.edges.index(edge)
        return self.num_of_nodes + index
        
    
    def calc_h_s(self):
        for vert1, _ in self.graph.adj_list[self.start]:
            if vert1 == self.start:
                continue
            index1 = self.edge_index((self.start, vert1))
            for vert2, _ in self.graph.adj_list[self.start]:
                if vert2 == self.start:
                    continue
                index2 = self.edge_index((self.start, vert2))
                if vert1 == vert2:
                    self.Q[index1][index2] += self.alpha
                else:
                    self.Q[index1][index2] += 2 * self.alpha
            self.Q[self.start][index1] -= 2 * self.alpha
            self.Q[index1][self.start] -= 2 * self.alpha
            
    def calc_h_t(self):
        for vert1, _ in self.graph.adj_list[self.end]:
            if vert1 == self.end:
                continue
            index1 = self.edge_index((self.end, vert1))
            for vert2, _ in self.graph.adj_list[self.end]:
                if vert2 == self.end:
                    continue
                index2 = self.edge_index((self.end, vert2))
                if vert1 == vert2:
                    self.Q[index1][index2] += self.alpha
                else:
                    self.Q[index1][index2] += 2 * self.alpha
            self.Q[self.end][index1] -= 2 * self.alpha
            self.Q[index1][self.end] -= 2 * self.alpha
            
    def calc_h_i(self, i):
        self.Q[i][i] += 4 * self.alpha
        for vert1, _ in self.graph.adj_list[i]:
            if vert1 == i:
                continue
            index1 = self.edge_index((i, vert1))
            for vert2, _ in self.graph.adj_list[i]:
                if vert2 == i:
                    continue
                index2 = self.edge_index((i, vert2))
                if vert1 == vert2:
                    self.Q[index1][index2] += self.alpha
                else:
                    self.Q[index1][index2] += 2 * self.alpha
            self.Q[i][index1] -= 4 * self.alpha
            self.Q[index1][i] -= 4 * self.alpha
            
    def calc_h_c(self):
        for edge in self.edges:
            index = self.edge_index(edge)
            vert1, vert2 = edge[0], edge[1]
            
            self.Q[index][index] += self.graph.connect_cost(vert1, vert2)
