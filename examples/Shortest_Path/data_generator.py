import networkx as nx
import random
import json
import os
from networkx.readwrite import json_graph


def generate_random_graph(min_nodes, max_nodes, min_edge_weight, max_edge_weight, p):
    while True:
        # Create a random Erdos Renyi graph
        G = nx.erdos_renyi_graph(random.randint(min_nodes, max_nodes), p)
        complement = list(nx.k_edge_augmentation(G, k=1, partial=True))
        G.add_edges_from(complement)
        nx.set_node_attributes(G, 0, 'src-tgt')
        nx.set_node_attributes(G, 0, 'sp')
        nx.set_node_attributes(G, 'node', 'entity')

        # Assign randomly weights to graph edges
        for (u, v, w) in G.edges(data=True):
            w['weight'] = random.randint(min_edge_weight, max_edge_weight)

        # Select a source and target nodes to compute the shortest path
        src, tgt = random.sample(list(G.nodes), 2)

        G.nodes[src]['src-tgt'] = 1
        G.nodes[tgt]['src-tgt'] = 1

        # Compute all the shortest paths between source and target nodes
        try:
            shortest_paths = list(nx.all_shortest_paths(G, source=src, target=tgt, weight='weight'))
        except:
            shortest_paths = []
        # Check if there exists only one shortest path
        if len(shortest_paths) == 1:
            for node in shortest_paths[0]:
                G.nodes[node]['sp'] = 1
            return nx.DiGraph(G)


def generate_dataset(file_name, num_samples, min_nodes=5, max_nodes=15, min_edge_weight=1, max_edge_weight=10, p=0.3):
    samples = []
    for _ in range(num_samples):
        G = generate_random_graph(min_nodes, max_nodes, min_edge_weight, max_edge_weight, p)
        G.remove_nodes_from([node for node, degree in dict(G.degree()).items() if degree == 0])
        samples.append(json_graph.node_link_data(G))

    with open(file_name, "w") as f:
        json.dump(samples, f)


root_dir = "./data"
if not os.path.exists(root_dir):
    os.makedirs(root_dir)
if not os.path.exists(root_dir + "/train"):
    os.makedirs(root_dir + "/train")
if not os.path.exists(root_dir + "/validation"):
    os.makedirs(root_dir + "/validation")

generate_dataset("./data/train/data.json", 1000)
generate_dataset("./data/validation/data.json", 100)
