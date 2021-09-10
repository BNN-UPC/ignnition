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
import argparse
from networkx.readwrite import json_graph
import json
import tarfile
import os
import sys


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


def migrate_dataset(input_path, output_path, max_per_file, split):
    gen = generator(input_path)
    data = []
    file_ctr_train = 0
    file_ctr_eval = 0

    tmp_dir = output_path+"/.tmp/"
    os.system("rm -rf %s" % (tmp_dir))
    os.makedirs(tmp_dir)

    counter = 0
    while True:
        try:
            G = next(gen)
            parser_graph = json_graph.node_link_data(G)
            data.append(parser_graph)
            if counter == max_per_file:
                a = np.random.rand()
                path = output_path 
                if a < split:
                    path += 'eval/'
                    with open(tmp_dir+'data.json', 'w') as json_file:
                        json.dump(data, json_file)

                    tar = tarfile.open(path + "sample_" + str(file_ctr_eval) + ".tar.gz", "w:gz")
                    tar.add(tmp_dir+'data.json')
                    tar.close()
                    os.remove(tmp_dir+'data.json')
                    file_ctr_eval += 1
                else:
                    path += 'train/'
                    with open(tmp_dir+'data.json', 'w') as json_file:
                        json.dump(data, json_file)

                    tar = tarfile.open(path + "sample_" + str(file_ctr_train) + ".tar.gz", "w:gz")
                    tar.add(tmp_dir+'data.json')
                    tar.close()
                    os.remove(tmp_dir+'data.json')

                    file_ctr_train += 1

                data = []
                counter = 0
            else:
                counter +=1

        # when finished, save all the remaining ones in the last file
        except:
            a = np.random.rand()
            path = output_path
            if a < split:
                path += 'eval/'
                with open(tmp_dir+'data.json', 'w') as json_file:
                    json.dump(data, json_file)

                tar = tarfile.open(path + "sample_" + str(file_ctr_eval) + ".tar.gz", "w:gz")
                tar.add(tmp_dir+'data.json')
                tar.close()
                os.remove(tmp_dir+'data.json')
            else:
                path += 'train/'
                with open(tmp_dir+'data.json', 'w') as json_file:
                    json.dump(data, json_file)

                tar = tarfile.open(path + "sample_" + str(file_ctr_train) + ".tar.gz", "w:gz")
                tar.add(tmp_dir+'data.json')
                tar.close()
                os.remove(tmp_dir+'data.json')
            os.system("rm -rf %s" % (tmp_dir))
            break

if __name__ == "__main__":
    # python migrate.py -d ../../../nsfnetbw/ -o ./datansfnet/ -n 100 -s 0.8
    # Parse logs and get best model
    parser = argparse.ArgumentParser(description='Parse file and create plots')

    parser.add_argument('-d', help='Origin data directory', type=str, required=True, nargs='+')
    parser.add_argument('-o', help='Output data directory', type=str, required=True, nargs='+')
    parser.add_argument('-n', help='Number of samples per file', type=int, required=True, nargs='+')
    parser.add_argument('-s', help='Percentage split of files used for TRAINING. 1-percentage will be added to EVALUATION set.', type=float, required=True, nargs='+')
    args = parser.parse_args()

    origin_dir = args.d[0]
    output_dir = args.o[0]
    split = 1-float(args.s[0])
    num_samples_file = args.n[0]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if os.path.exists(output_dir+'/eval'):
        os.system("rm -rf %s" % (output_dir))
    
    if os.path.exists(output_dir+'/train'):
        os.system("rm -rf %s" % (output_dir))

    os.makedirs(output_dir+'/eval')
    os.makedirs(output_dir+'/train')

    migrate_dataset(origin_dir, output_dir, num_samples_file, split)