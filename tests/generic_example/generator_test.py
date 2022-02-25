import ignnition
import networkx as nx
import random
import os
from networkx.readwrite import json_graph
import random
from itertools import combinations, groupby


def random_connected_graph(n, p):
    edges = combinations(range(n), 2)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    if p <= 0:
        return G
    if p >= 1:
        return nx.complete_graph(n, create_using=G)
    for _, node_edges in groupby(edges, key=lambda x: x[0]):
        node_edges = list(node_edges)
        random_edge = random.choice(node_edges)
        G.add_edge(*random_edge)
        for e in node_edges:
            if random.random() < p:
                G.add_edge(*e)
    return G


def assign_random_entities(G, entity_number=2, features_per_entity=None):
    if features_per_entity is None:
        features_per_entity = [3, 2]
    for node in G.nodes():
        entity = random.randint(0, entity_number - 1)
        G.nodes[node]['entity'] = f'entity{entity}'
        rand_sum = 0
        for feature in range(features_per_entity[entity]):
            rand_num = random.random()
            G.nodes[node][f'feature{feature + sum(features_per_entity[:entity])}'] = random.random()
            rand_sum += rand_num
        G.nodes[node][f'label{entity}'] = rand_sum
    return G


def generate_dataset(num_samples=5, min_nodes=20, max_nodes=30, p=0.3):
    samples = []
    for _ in range(num_samples):
        G = random_connected_graph(random.randint(min_nodes, max_nodes), p)
        G = assign_random_entities(G)
        yield G.to_directed()


def main():
    model = ignnition.create_model(model_dir='./working/baseline',
                                   training_data=generate_dataset())
    model.computational_graph()
    model.train_and_validate()


main()