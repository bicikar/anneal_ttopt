from src.field import Field

def hop_path_results(hop_result, field, shortest, min_length):
    cost = 0
    admissible = False
    if len(hop_result) == field.hop_based.P and len({x[1] for x in hop_result}) == field.hop_based.P:
        admissible = True
        
    path_correctness = True
    
    path = []
    for node, pos in hop_result:
        if node not in path:
            path.append(node)
            
    identical_vertices = False
    if set(shortest) == set(path):
        identical_vertices = True
        
    for i in range(len(path) - 1):
        cur_cost = field.graph.connect_cost(field.get_index(path[i][1], path[i][0]), field.get_index(path[i+1][1], path[i+1][0]))
        if cur_cost < 0:
            path_correctness = False
            break
        cost += cur_cost
    
    if cost >= min_length:
        return path_correctness, admissible, identical_vertices, cost / min_length
    else:
        return False, False, identical_vertices, 0


def find_edge_index(arr, node):
    one_start = False
    n_index = -1
    for index, el in enumerate(arr):
        if node in el:
            if one_start:
                return -1
            n_index = index
            one_start = True
    return n_index


def get_second_vertice(pair, vert):
    if vert == pair[0]:
        return pair[1]
    return pair[0]

def edge_path_results(edge_result, field, min_length):
    cost = 0
    path = []
    admissible = False
    path_correctness = True
    
    node = (field.start[1], field.start[0]);
    n_index = find_edge_index(edge_result, node)
    if n_index == -1:
        return False, False, path, 0
    
    next_node = get_second_vertice(edge_result[n_index], node)
    edge_result.pop(n_index)
    cost += field.graph.connect_cost(field.get_index(node[1], node[0]), field.get_index(next_node[1], next_node[0]))
    path.append(node)
    path.append(next_node)
    end_found = False
    
    while len(edge_result):
        n_index = find_edge_index(edge_result, next_node)
        if n_index == -1:
            return False, False, path, cost / min_length
        node = next_node
        next_node = get_second_vertice(edge_result[n_index], node)
        edge_result.pop(n_index)
        cur_cost = field.graph.connect_cost(field.get_index(node[1], node[0]), field.get_index(next_node[1], next_node[0]))
        cost += cur_cost
        path.append(next_node)
        if next_node == (field.end[1], field.end[0]):
            end_found = True
            break
    if not end_found:
        return False, False, path, cost / min_length
    if not len(edge_result):
        admissible = True
    if cost >= min_length:
        return path_correctness, admissible, path, cost / min_length
    else:
        return False, False, path, 0

def run_stats(field, calls, hop_iter, edge_iter, iters=100, graph_num=1):
    correct_path_ttopt_edge = 0
    correct_path_ttopt_hop = 0
    correct_path_edge_init = 0
    correct_path_edge = 0
    correct_path_hop_init = 0
    correct_path_hop = 0

    admissible_ttopt_edge = 0
    admissible_ttopt_hop = 0
    admissible_edge_init = 0
    admissible_edge = 0
    admissible_hop_init = 0
    admissible_hop = 0

    identical_hop_to_shrotest = 0
    identical_hop_init_to_shrotest = 0
    identical_ttopt_hop_to_shrotest = 0

    min_cost, shortest = field.calculate_shortest()
    edge_init_costs = []
    edge_costs = []
    ttopt_edge_costs = []
    ttopt_hop_costs = []
    hop_costs = []
    hop_init_costs = []
    true_shortest = [0, 0, 0, 0, 0, 0]
    
    for gi in range(graph_num):
        print(f'Running graph num{gi}')
        for i in range(iters):
            if i % 5 == 0:
                print("Going through iter", i)
            initial_edge, ttopt_edge_edges = field.prepare_initial("edge_based", calls)
            initial_hop, ttopt_hop_edges = field.prepare_initial("hop_based", calls)

            edge_init_res = field.evaluate_annealing("edge_based", 100, edge_iter, initial_edge)
            edge_res = field.evaluate_annealing("edge_based", 100, edge_iter)
            hop_res = field.evaluate_annealing("hop_based", 100, hop_iter)
            hop_init_res = field.evaluate_annealing("hop_based", 100, hop_iter, initial_hop)

            ttopt_edge_info = edge_path_results(ttopt_edge_edges, field, min_cost)
            hop_info = hop_path_results(hop_res, field, shortest, min_cost)
            hop_init_info = hop_path_results(hop_init_res, field, shortest, min_cost)
            ttopt_hop_info = hop_path_results(ttopt_hop_edges, field, shortest, min_cost)
            edge_init_info = edge_path_results(edge_init_res, field, min_cost)
            edge_info = edge_path_results(edge_res, field, min_cost)

            if edge_init_info[0]:
                correct_path_edge_init += 1
                edge_init_costs.append(edge_init_info[3])
                if edge_init_info[3] == 1.0:
                    true_shortest[0] += 1
            if edge_init_info[1]:
                admissible_edge_init += 1

            if hop_init_info[0]:
                correct_path_hop_init += 1
                hop_init_costs.append(hop_init_info[3])
                if hop_init_info[3] == 1.0:
                    true_shortest[1] += 1
            if hop_init_info[1]:
                admissible_hop_init += 1
            if hop_init_info[2]:
                identical_hop_init_to_shrotest += 1

            if edge_info[0]:
                correct_path_edge += 1
                edge_costs.append(edge_info[3])
                if edge_info[3] == 1.0:
                    true_shortest[2] += 1
            if edge_info[1]:
                admissible_edge += 1

            if hop_info[0]:
                correct_path_hop += 1
                hop_costs.append(hop_info[3])
                if hop_info[3] == 1.0:
                    true_shortest[3] += 1
            if hop_info[1]:
                admissible_hop += 1
            if hop_info[2]:
                identical_hop_to_shrotest += 1

            if ttopt_edge_info[0]:
                correct_path_ttopt_edge += 1
                ttopt_edge_costs.append(ttopt_edge_info[3])
                if ttopt_edge_info[3] == 1.0:
                    true_shortest[4] += 1
            if ttopt_edge_info[1]:
                admissible_ttopt_edge += 1

            if ttopt_hop_info[0]:
                correct_path_ttopt_hop += 1
                ttopt_hop_costs.append(ttopt_hop_info[3])
                if ttopt_hop_info[3] == 1.0:
                    true_shortest[5] += 1
            if ttopt_hop_info[1]:
                admissible_ttopt_hop += 1
            if ttopt_hop_info[2]:
                identical_ttopt_hop_to_shrotest += 1
                
        field = Field(field.width, field.height, end=(0, 0))
        min_cost, shortest = field.calculate_shortest()

    correct_path = {}
    correct_path["ttopt_edge"] = correct_path_ttopt_edge
    correct_path["ttopt_hop"] = correct_path_ttopt_hop
    correct_path["edge_init"] = correct_path_edge_init
    correct_path["edge"] = correct_path_edge
    correct_path["hop_init"] = correct_path_hop_init
    correct_path["hop"] = correct_path_hop
    
    admissible = {}
    admissible["ttopt_edge"] = admissible_ttopt_edge
    admissible["ttopt_hop"] = admissible_ttopt_hop
    admissible["edge_init"] = admissible_edge_init
    admissible["edge"] = admissible_edge
    admissible["hop_init"] = admissible_hop_init
    admissible["hop"] = admissible_hop
    
    identical = {}
    identical["hop"] = identical_hop_to_shrotest
    identical["hop_init"] = identical_hop_init_to_shrotest
    identical["ttopt_hop"] = identical_ttopt_hop_to_shrotest
    
    costs = {}
    costs["ttopt_edge"] = ttopt_edge_costs
    costs["ttopt_hop"] = ttopt_hop_costs
    costs["edge_init"] = edge_init_costs
    costs["edge"] = edge_costs
    costs["hop_init"] = hop_init_costs
    costs["hop"] = hop_costs
    
    return correct_path, admissible, identical, costs, true_shortest