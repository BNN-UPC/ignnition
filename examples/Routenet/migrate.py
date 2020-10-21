'''
 *
 * Copyright (C) 2020 Universitat Politècnica de Catalunya.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
'''

import numpy as np
import networkx as nx
import os
import random
import json
import sys
import argparse
import datanetAPI
import tarfile



def get_corresponding_values(sample, avgBws_lst, dly_lst, jit_lst):
    it = 0
    netSize = sample.get_network_size()
    for i in range(netSize):
        for j in range(netSize):
            if i!=j:
                avgBws_lst[it] = sample.get_srcdst_traffic(i,j)["AggInfo"]["AvgBw"]
                dly_lst[it] = sample.get_srcdst_performance(i,j)["AggInfo"]["AvgDelay"]
                jit_lst[it] = sample.get_srcdst_performance(i,j)["AggInfo"]["Jitter"]
                it += 1

def get_link_capacities_lst(sample):
    link_capacities = []
    netSize = sample.get_network_size()
    for i in range(netSize):
        for j in range(netSize):
            if i!=j:
                bw = int(sample.get_srcdst_link_bandwidth(i,j))
                if (bw != -1):
                    link_capacities.append(bw)
    return(link_capacities)


def process_sample(sample):
    
    G = sample.get_topology_object()
    netSize = sample.get_network_size()
    
    n_paths = netSize*(netSize-1)
    
    traffic_lst = np.zeros(n_paths)
    dly_lst = np.zeros(n_paths)
    jit_lst = np.zeros(n_paths)
    
    get_corresponding_values(sample, traffic_lst, dly_lst, jit_lst)
    link_capacities = get_link_capacities_lst (sample)
    
    data = {"traffic": list(traffic_lst), "delay": list(dly_lst),
                "jitter": list(jit_lst), "link_capacity": list(link_capacities)}
    
    dict_links = {}

    data['link'] = []

    edge_index = 0
    for e in G.edges:
        link_name = "l" + str(edge_index)
        data['link'].append(link_name)
        dict_links["%d-%d" % (e[0],e[1])] = link_name
        edge_index += 1

    #here we create the adjcecencies
    data['adj_paths_links'] = {}
    data['adj_links_paths'] = {}

    data['path'] = []
    path_index =  0
    for i in range(netSize):
        for j in range(netSize):
            if (i == j):
                continue
            path_name = "p" + str(path_index)
            data['path'].append(path_name)

            path = sample.get_srcdst_routing(i,j)
            i0 = 0
            for i1 in range(1,len(path)):
                link_name = dict_links["%d-%d" % (path[i0],path[i1])]
                if not link_name in data['adj_paths_links'].keys():
                    data['adj_paths_links'][link_name] = []
                if not path_name in data['adj_links_paths'].keys():
                    data['adj_links_paths'][path_name] = []
                
                data['adj_paths_links'][link_name].append(path_name)
                data['adj_links_paths'][path_name].append(link_name)
                
                i0 = i1
                
            path_index += 1
    return data

class SampleLst:
    def __init__(self,path):
        self.sample_lst = []
        self.file_ctr = 0
        self.path = path
        if not os.path.exists(path):
            os.makedirs(path)
        
    def saveJson(self):
        with open('data.json', 'w') as outfile:
            json.dump(self.sample_lst, outfile)
        tar = tarfile.open(self.path + "/sample_" + str(self.file_ctr) + ".tar.gz", "w:gz")
        tar.add('data.json')
        tar.close()
        os.remove('data.json')
        self.file_ctr += 1
        self.sample_lst.clear()


def data(args): 
    directory_path = args.dataset
    output_path = args.output_path
    
    samples_per_file = 100
    train_struct = SampleLst(output_path + 'Dataset_routenet/train')
    eval_struct = SampleLst(output_path + 'Dataset_routenet/eval')
    
    
    dataset = datanetAPI.DatanetAPI(directory_path)
    it = iter(dataset)
    for sample in it:
        x = random.uniform(0, 1)
        if x <= 0.8:
            samples_struct = train_struct
        else:
            samples_struct = eval_struct
        
        samples_struct.sample_lst.append(process_sample(sample))
        if (len(samples_struct.sample_lst) == samples_per_file):
            samples_struct.saveJson()
    
    if (len(train_struct.sample_lst) != 0):
        train_struct.saveJson()
    
    if (len(eval_struct.sample_lst) != 0):
        train_struct.saveJson()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Migrating tool from a raw dataset to a valid JSON version.')

    parser.add_argument('--dataset', type=str,help='Path to find the dataset')
    parser.add_argument('--output_path', help='Path where the resulting JSON dataset will be saved', type=str)
    parser.set_defaults(func=data)

    args = parser.parse_args()
    args.func(args)

