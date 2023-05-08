from qubovert.sim import anneal_qubo
from qubovert import boolean_var
from anneals import Graph, HopBased, EdgeBased
from ttopt import TTOpt
import numpy as np
import matplotlib.pyplot as plt
import math


def norm_noise():
    res = np.random.normal()
    if res < 0:
        return max(-2, res)
    return min(2, res)

'''
Builds a field with width=w and height=h and a graph on corresponding field.
Calculates Q with hop-based algorithm.
'''
class Field:
    def __init__(self, w, h, speed=20, end=(0, 0), start=None, graph=None):
        self.width = w
        self.height = h
        self.speed = speed
        if start is None:
            self.start = (w // 2, h - 1)
        else:
            self.start = start
        self.end = end
        self.min_speed = 2
        if graph is None:
            self.blocks = self.define_speed()
            self.graph = self.build_graph()
        else:
            self.graph = graph
        print("Graph build")
        self.hop_based = self.build_hop_based()
        print("Hop based build")
        self.edge_based = self.build_edge_based()
        print("Edge based build")
        
    def define_speed(self):
        blocks = [[0 for row in range(self.width)] for col in range(self.height)]
        coef = (self.speed - self.min_speed) / (self.width // 2 + self.height - 1)
        for h in range(self.height):
            for w in range(self.width):
                blocks[h][w] = self.speed - coef * (abs(w - self.start[0]) + abs(h - self.start[1])) + \
                norm_noise()
        return blocks
    
    def draw(self):
        string_array = []
        for row in self.blocks:
            string_array.append(list(map(lambda x: "%.2f" % x, row)))
        colors = [["w" for row in range(self.width)] for col in range(self.height)]
        colors[self.end[1]][self.end[0]] = "#64c471"
        colors[self.start[1]][self.start[0]] = "#c74444"
        fig, ax = plt.subplots()
        ax.set_axis_off()
        table = ax.table(cellText=string_array, loc='center', cellColours=colors)
        table.scale(1, 2)
        table.set_fontsize(16)
        fig.tight_layout()
        plt.show()
        
        
    def get_pos(self, index):
        return index // self.width, index % self.width
    
    def get_index(self, w, h):
        return h * self.width + w
    
    def get_neighbours(self, ind):
        dh = [-self.width, 0, self.width]
        dw = [-1, 0, 1]
        neighbours = []
        for h in dh:
            for w in dw:
                if h == 0 and w == 0:
                    continue
                new_ind = ind + h + w
                if new_ind >= 0 and new_ind < self.width * self.height and new_ind // self.width == (ind + h) // self.width:
                    neighbours.append(new_ind)
        return neighbours
    
    def build_graph(self):
        graph = Graph(self.width * self.height)
        for ind in range(self.width * self.height):
            neighbours = self.get_neighbours(ind)
            for n in neighbours:
                if graph.connect_cost(ind, n) == -1:
                    ind_pos = self.get_pos(ind)
                    n_pos = self.get_pos(n)
                    cost = self.blocks[ind_pos[0]][ind_pos[1]] + self.blocks[n_pos[0]][n_pos[1]]
                    graph.add_edge(ind, n, cost / 2)
        return graph
    
    def calculate_shortest(self):
        d = [math.inf for i in range(self.graph.num_of_nodes)]
        used = [False for i in range(self.graph.num_of_nodes)]
        parent = [0 for i in range(self.graph.num_of_nodes)]
        start = self.get_index(self.start[0], self.start[1])
        d[start] = 0
        for i in range(self.graph.num_of_nodes):
            v = -1
            for j in range(self.graph.num_of_nodes):
                if not used[j] and (v == -1 or d[j] < d[v]):
                    v = j
            if d[v] == math.inf:
                break
            used[v] = True
            for (e, cost) in self.graph.adj_list[v]:
                if d[v] + cost < d[e]:
                    d[e] = d[v] + cost
                    parent[e] = v
#         print("Shortest length:", d[0])
        path = []
        node = self.get_index(self.end[0], self.end[1])
        
        while node != start:
            path.append(self.get_pos(node))
            node = parent[node]
        path.append(self.get_pos(start))
        path.reverse()
#         print("Shortest path:", path)
        if d[0] != 0:
            return d[0], path
        else:
            return d[-1], path
    
    def build_hop_based(self, p=None):
        if p is None:
            p = self.width // 2 + self.height
        hop_based = HopBased(p, self.graph,
                             start=self.get_index(self.start[0], self.start[1]), 
                             end=self.get_index(self.end[0], self.end[1]))
        hop_based.calc_h_p()
        hop_based.calc_h_c()
        hop_based.calc_h_s_t()
        
        for i in range(len(hop_based.Q)):
            for j in range(len(hop_based.Q)):
                if i > j:
                    hop_based.Q[i][j] = 0
        return hop_based
    
    def build_edge_based(self):
        edge_based = EdgeBased(self.graph, start=self.get_index(self.start[0], self.start[1]), 
                             end=self.get_index(self.end[0], self.end[1]))
        edge_based.calc_h_s()
        edge_based.calc_h_t()
        for i in range(edge_based.num_of_nodes):
            if i != edge_based.start and i != edge_based.end:
                edge_based.calc_h_i(i)
        edge_based.calc_h_c()
        
        for i in range(len(edge_based.Q)):
            for j in range(len(edge_based.Q)):
                if i > j:
                    edge_based.Q[i][j] = 0
        return edge_based 
        

    def prepare_initial(self, algorithm, num_calls):
        initial = {}
        if algorithm == "edge_based":
            Q_matr = np.array(self.edge_based.Q)
            
            def f(X):  
                res = []
                for row in X:
                    res = np.append(res, np.dot(row, np.dot(Q_matr, row)))# Target function

                return res
            tto = TTOpt(
                f=f,                    # Function for minimization. X is [samples, dim]
                d=len(Q_matr),                    # Number of function dimensions
                a=0.,                 # Grid lower bound (number or list of len d)
                b=1.,                 # Grid upper bound (number or list of len d)
                n=2,                 # Number of grid points (number or list of len d)
                evals=num_calls,            # Number of function evaluations
                )
            
            tto.minimize(15)
            x = tto.x_min
            
            res_ttopt = []
            for i, val in enumerate(x[self.edge_based.num_of_nodes:]):
                edge = self.edge_based.edges[i]
                if val == 1:
                    res_ttopt.append((self.get_pos(edge[0]), self.get_pos(edge[1])))
                    
            initial = {}
            N = self.edge_based.num_of_nodes + self.edge_based.num_of_edges
            for i in range(N):
                if i < self.edge_based.num_of_nodes:
                    initial[self.get_pos(i)] = x[i]
                else:
                    cur_edge = self.edge_based.edges[i - self.edge_based.num_of_nodes]
                    initial[(-100, self.get_pos(cur_edge[0]), self.get_pos(cur_edge[1]))] = x[i]

        else:
            Q_matr = np.array(self.hop_based.Q)
            
            def f(X):  
                res = []
                for row in X:
                    res = np.append(res, np.dot(row, np.dot(Q_matr, row)))# Target function

                return res
            tto = TTOpt(
                f=f,                    # Function for minimization. X is [samples, dim]
                d=len(Q_matr),                    # Number of function dimensions
                a=0.,                 # Grid lower bound (number or list of len d)
                b=1.,                 # Grid upper bound (number or list of len d)
                n=2,                 # Number of grid points (number or list of len d)
                evals=num_calls,            # Number of function evaluations
                )
            
            tto.minimize(15)
            x = tto.x_min
            
            res_ttopt = []
            for i, val in enumerate(x):
                if val == 1:
                    res_ttopt.append((self.get_pos(i // self.hop_based.P), i % self.hop_based.P))
            res_ttopt = sorted(res_ttopt, key=lambda x: x[1])
            initial = {}
            for i in range(len(x)):
                initial[(self.get_pos(i // self.hop_based.P), i % self.hop_based.P)] = x[i]

        return initial, res_ttopt
            
        
    def evaluate_annealing(self, algorithm, number=1, duration=1000, initial=None):
        if algorithm == "hop_based":
            N = self.hop_based.P * self.hop_based.num_of_nodes

            x = {i: boolean_var((self.get_pos(i // self.hop_based.P), i % self.hop_based.P)) for i in range(N)}

            model = 0
            for x1 in range(N):
                for x2 in range(x1, N):
                    model += self.hop_based.Q[x1][x2] * x[x1] * x[x2]
        if algorithm == "edge_based":
            N = self.edge_based.num_of_edges + self.edge_based.num_of_nodes
            x = {}
            for i in range(N):
                if i < self.edge_based.num_of_nodes:
                    x[i] = boolean_var(self.get_pos(i))
                else:
                    cur_edge = self.edge_based.edges[i - self.edge_based.num_of_nodes]
                    x[i] = boolean_var((-100, self.get_pos(cur_edge[0]), self.get_pos(cur_edge[1])))
#             x = {i: boolean_var(self.get_pos(i) if i < self.edge_based.num_of_nodes else (-100, self.edge_based.edges[i - self.edge_based.num_of_nodes])) for i in range(N)}

            model = 0
            for x1 in range(N):
                for x2 in range(x1, N):
                    model += self.edge_based.Q[x1][x2] * x[x1] * x[x2]
        
        if initial is None:
            res = anneal_qubo(model, num_anneals=number, anneal_duration=duration)
        else:
            res = anneal_qubo(model, num_anneals=number, anneal_duration=duration, initial_state=initial)
#         print("STEPS:")
#         for rs in res:
#             print(rs.value, rs.state)
        result = res.best.state
        if algorithm == "edge_based":
            zero_keys = [key for key in result if result[key] == 0 or type(key[1]) == int]
        else:
            zero_keys = [key for key in result if result[key] == 0]

        for key in zero_keys:
            result.pop(key)
        if algorithm == "hop_based":
            res_nodes = list(sorted(result.keys(), key=lambda x: x[1]))
#             res_nodes = []

#             for node, pos in keys_sorted:
#                 if node not in res_nodes:
#                     res_nodes.append(node)

#             print("Path in P nodes: ", keys_sorted)
#             print("Real path: ", res_nodes)
        else:
            res_nodes = list(reversed([(key[1], key[2]) for key, _ in result.items()]))
#             print("Real path:", list(reversed(res_nodes)))
        return res_nodes