import matplotlib.pyplot as plt

def draw_table_stat(correct_path, true_shortest, admissible, identical):
    N = 100
    string_array = [["",
                     "Edge init annealing", 
                     "Hop init annealing", 
                     "Edge classic annealing", 
                     "Hop classic annealing", 
                     "TTopt on edge", 
                     "TTopt on hop"]]
    string_array.append(["Correct path", 
                     "{} / {:.2f}%".format(correct_path["edge_init"], 100 * correct_path["edge_init"] / N),
                     "{} / {:.2f}%".format(correct_path["hop_init"], 100 * correct_path["hop_init"] / N),
                     "{} / {:.2f}%".format(correct_path["edge"], 100 * correct_path["edge"] / N), 
                     "{} / {:.2f}%".format(correct_path["hop"], 100 * correct_path["hop"] / N), 
                     "{} / {:.2f}%".format(correct_path["ttopt_edge"], 100 * correct_path["ttopt_edge"] / N),
                     "{} / {:.2f}%".format(correct_path["ttopt_hop"], 100 * correct_path["ttopt_hop"] / N)])
    string_array.append(["True shortest", 
                         "{} / {:.2f}%".format(true_shortest[0], 100 * true_shortest[0] / N), 
                         "{} / {:.2f}%".format(true_shortest[1], 100 * true_shortest[1] / N),
                         "{} / {:.2f}%".format(true_shortest[2], 100 * true_shortest[2] / N),
                         "{} / {:.2f}%".format(true_shortest[3], 100 * true_shortest[3] / N),
                         "{} / {:.2f}%".format(true_shortest[4], 100 * true_shortest[4] / N), 
                         "{} / {:.2f}%".format(true_shortest[5], 100 * true_shortest[5] / N)])
    string_array.append(["Admissible path", 
                         "{} / {:.2f}%".format(admissible["edge_init"], 100 * admissible["edge_init"] / N), 
                         "{} / {:.2f}%".format(admissible["hop_init"], 100 * admissible["hop_init"] / N),
                         "{} / {:.2f}%".format(admissible["edge"], 100 * admissible["edge"] / N),
                         "{} / {:.2f}%".format(admissible["hop"], 100 * admissible["hop"] / N),
                         "{} / {:.2f}%".format(admissible["ttopt_edge"], 100 * admissible["ttopt_edge"] / N), 
                         "{} / {:.2f}%".format(admissible["ttopt_hop"], 100 * admissible["ttopt_hop"] / N)])
    string_array.append(["Equal vertices", 
                         "-",
                         "{} / {:.2f}%".format(identical["hop_init"], 100 * identical["hop_init"] / N),
                         "-",
                         "{} / {:.2f}%".format(identical["hop"], 100 * identical["hop"] / N),
                         "-",
                         "{} / {:.2f}%".format(identical["ttopt_hop"], 100 * identical["ttopt_hop"] / N)])
    fig, ax = plt.subplots()
    ax.set_axis_off()
    table = ax.table(cellText=string_array, loc='center')
    table.scale(3, 3)
    table.set_fontsize(24)
    fig.tight_layout()
    plt.show()
    
def draw_big_boxplot(costs):
    fig = plt.figure(figsize =(8, 5))
    fig.suptitle('Path length')
    ax = fig.add_axes([0, 0, 1, 1])

    labels = ["Edge init", "Hop init", "Edge classic", "Hop classic", "TTopt on edge", "TTopt on hop"]
    bp = ax.boxplot([costs["edge_init"],
                     costs["hop_init"], 
                     costs["edge"], 
                     costs["hop"], 
                     costs["ttopt_edge"], 
                     costs["ttopt_hop"]], labels=labels)
    plt.show()
    
def draw_medium_boxplot(costs):
    fig = plt.figure(figsize =(8, 5))
    fig.suptitle('Path length')
    ax = fig.add_axes([0, 0, 1, 1])

    labels = ["Edge classic", "Hop classic", "TTopt on edge", "TTopt on hop"]
    bp = ax.boxplot([costs["edge"], 
                     costs["hop"], 
                     costs["ttopt_edge"], 
                     costs["ttopt_hop"]], labels=labels)
    plt.show()

def draw_small_boxplot(costs):
    fig = plt.figure(figsize =(8, 5))
    fig.suptitle('Path length')
    ax = fig.add_axes([0, 0, 1, 1])

    labels = ["Edge classic", "Hop classic"]
    bp = ax.boxplot([costs["edge"], 
                     costs["hop"]], labels=labels)
    plt.show()