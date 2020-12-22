"""
   Copyright 2020 Universitat Politècnica de Catalunya

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import numpy as np
import tensorflow as tf
import random
import networkx as nx
from datanetAPI import DatanetAPI


def generator(data_dir):
    tool = DatanetAPI(data_dir)
    it = iter(tool)
    for sample in it:
        G_copy = sample.get_topology_object().copy()
        T = sample.get_traffic_matrix()
        R = sample.get_routing_matrix()
        D = sample.get_performance_matrix()

        HG = network_to_hypergraph(network_graph=G_copy,
                                   routing_matrix=R,
                                   traffic_matrix=T,
                                   performance_matrix=D)
        print(HG)
        yield HG


def network_to_hypergraph(network_graph, routing_matrix, traffic_matrix, performance_matrix):
    G = (nx.DiGraph(network_graph)).copy()
    R = np.copy(routing_matrix)
    T = np.copy(traffic_matrix)
    P = np.copy(performance_matrix)

    D_G = nx.DiGraph()
    for src in range(G.number_of_nodes()):
        for dst in range(G.number_of_nodes()):
            if src != dst:
                D_G.add_node('p_{}_{}'.format(src, dst),
                             entity='path',
                             traffic=T[src, dst]['Flows'][0]['AvgBw'],
                             delay=P[src, dst]['AggInfo']['AvgDelay'])

                if G.has_edge(src, dst):
                    D_G.add_node('l_{}_{}'.format(src, dst),
                                 entity='link',
                                 capacity=int(G.edges[src, dst]['bandwidth']))

                for h_1, h_2 in [R[src, dst][i:i + 2] for i in range(0, len(R[src, dst]) - 1)]:
                    D_G.add_edge('p_{}_{}'.format(src, dst), 'l_{}_{}'.format(h_1, h_2))
                    D_G.add_edge('l_{}_{}'.format(h_1, h_2), 'p_{}_{}'.format(src, dst))

    D_G.remove_nodes_from([node for node, out_degree in D_G.out_degree() if out_degree == 0])

    return D_G

import matplotlib.pyplot as plt
from networkx.readwrite import json_graph
import json
import tarfile
import os
def migrate_dataset(input_path, output_path, max_per_file):
    gen = generator(input_path)
    data = []
    file_ctr = 0
    counter = 0
    while True:
        try:
            G = next(gen)
            parser_graph = json_graph.node_link_data(G)
            data.append(parser_graph)
            if counter == max_per_file:
                a = np.random.rand()
                path = output_path + 'Dataset_routenet_networkx/'
                if a < 0.2:
                    path += 'EVAL/'
                else:
                    path += 'TRAIN/'

                with open('data.json', 'w') as json_file:
                        json.dump(data, json_file)

                tar = tarfile.open(path + "/sample_" + str(file_ctr) + ".tar.gz", "w:gz")
                tar.add('data.json')
                tar.close()
                os.remove('data.json')

                data = []
                counter = 0
                file_ctr += 1
            else:
                counter +=1

            #visualize the plot
            #plt.subplot(121)
            #nx.draw(G, with_labels=True, font_weight='bold')
            #plt.show()

        #when finished, save all the remaining ones
        except:
            a = np.random.rand()
            path = output_path + 'Dataset_routenet_networkx/'
            if a < 0.2:
                path += 'EVAL/'
            else:
                path += 'TRAIN/'

            with open(path + 'data.json', 'w') as json_file:
                json.dump(data, json_file)

            tar = tarfile.open(path + "/sample_" + str(file_ctr) + ".tar.gz", "w:gz")
            tar.add(path + 'data.json')
            tar.close()
            os.remove(path + 'data.json')



migrate_dataset('/Users/david/Documents/BNN/Datasets/nsfnetbw', output_path='./', max_per_file=100)