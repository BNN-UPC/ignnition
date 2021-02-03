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

# -*- coding: utf-8 -*-

import glob
import json
import sys
import tarfile
import numpy as np
import math
import random
import tensorflow as tf
import json
import warnings
from ignnition.utils import *

import networkx as nx
from networkx.readwrite import json_graph


class Generator:
    """
    This class implements the Generator in charge of feeding the data to the main GNN module. This class will take as input the original datasets of the user or the passed array and compute a series of transformation and precalculations. Finally it serves it to the GNN module.

    Attributes
    ----------
    end_symbol:    str
        End symbol to be used when reading the json file as a stream of data.

    Methods:
    ----------
    stream_read_json(self, f)
       Creates a generator of samples from the given dataset or sample array. All these are read as a stream of data to avoid full alocation on memory.

    __process_sample(self, sample, file=None)
        Given an input sample, it processes it and pre-computes several aspects to be later served to the GNN module.

    generate_from_array
        Creates and returns the generator from an input array of samples of the user.

    generate_from_dataset
        Creates and returns the generator from an input dataset of samples of the user.
    """

    def __init__(self):
        self.end_symbol = bytes(']', 'utf-8')

    def stream_read_json(self, f):
        """
        Parameters
        ----------
        f:    dict
            Input data
        """

        start_pos = 1
        while True:
            try:
                obj = json.load(f)
                yield obj
                return
            except json.JSONDecodeError as e:
                f.seek(start_pos)
                json_str = f.read(e.pos)
                obj = json.loads(json_str)
                start_pos += e.pos + 1
                a = f.read(1)  # this 1 is the coma or the final symbol

                if a == self.end_symbol or a == ']':
                    yield obj
                    return
                yield obj

    def __process_sample(self, sample, file=None):
        """
        Parameters
        ----------
        sample:    dict
            Input sample which is a serialized version (in JSON) of a networkx graph.
        file:    str
            Path to these file (which is useful for error-checking purposes)
        """

        # load the model
        G = json_graph.node_link_graph(sample)

        entity_counter = {}
        mapping = {}
        data = {}

        for name in self.entity_names:
            entity_counter[name] = 0

        list_nodes = list(G.nodes())
        for node_name in list_nodes:
            attributes = G.nodes[node_name]

            if 'entity' not in attributes:
                print_failure("Error in the dataset file located in '" + file + ".")
                print_failure('The node named' + node_name + '" was not assigned an entity.')

            entity_name = attributes['entity']
            new_node_name = entity_name + '_{}'
            num_node = entity_counter[entity_name]
            entity_counter[entity_name] += 1

            mapping[node_name] = new_node_name.format(num_node)

        # save the number of nodes of each entity
        for name in self.entity_names:
            data['num_' + name] = entity_counter[name]

        # do we need this??
        D_G = nx.relabel_nodes(G, mapping)

        # load the features (all the features are set to be lists. So we always return a list of lists)
        for f in self.feature_names:
            try:
                feature = np.array(list(nx.get_node_attributes(D_G, f).values()))

                # it should always be a 2d array
                if len(np.shape(feature)) == 1:
                    feature = np.expand_dims(feature, axis=-1)

                if feature.size == 0:
                    message = "The feature " + f + " was used in the model_description.yaml file " \
                                                   "but was not defined in the dataset."
                    if file is not None:
                        message = "Error in the dataset file located in '" + file + ".\n" + message
                    print_failure(message)
                else:
                    data[f] = feature

            except:
                message = "The feature " + f + " was used in the model_description.yaml file " \
                                               "but was not defined in the dataset."
                if file is not None:
                    message = "Error in the dataset file located in '" + file + ".\n" + message
                print_failure(message)

        # take other inputs if needed (check that they might be global features)
        for a in self.additional_input:
            node_attr = np.array(list(nx.get_node_attributes(D_G, a).values()))
            # it should always be a 2d array
            if len(np.shape(node_attr)) == 1:
                node_attr = np.expand_dims(node_attr, axis=-1)

            edge_attr = np.array(list(nx.get_edge_attributes(D_G, a).values()))
            # it should always be a 2d array
            if len(np.shape(edge_attr)) == 1:
                edge_attr = np.expand_dims(edge_attr, axis=-1)

            if node_attr.size != 0:
                data[a] = node_attr
            elif edge_attr.size != 0:
                data[a] = edge_attr
            elif a in D_G.graph:
                data[a] = [D_G.graph[a]]
            else:
                message = 'The data named "' + a + '" was used in the model_description.yaml file ' \
                                                   'but was not defined in the dataset.'
                if file is not None:
                    message = "Error in the dataset file located in '" + file + ".\n" + message
                print_failure(message)

        if self.training:
            # collect the output
            try:
                node_output = list(nx.get_node_attributes(D_G, self.output_name).values())
                final_output = node_output
            except:
                global_output = list(D_G.graph[self.output_name].values())
                final_output = global_output

        # find the adjacencies
        edges_list = list(D_G.edges())
        processed_neighbours = {}
        for e in edges_list:
            src_node, dst_node = e
            src_num = int(src_node.split('_')[-1])
            dst_num = int(dst_node.split('_')[-1])
            src_entity = D_G.nodes[src_node]['entity']
            dst_entity = D_G.nodes[dst_node]['entity']

            if dst_node not in processed_neighbours:
                processed_neighbours[dst_node] = 0

            # all the necessary info for the adjacency lists
            if 'src_' + src_entity + '_to_' + dst_entity not in data:
                data['src_' + src_entity + '_to_' + dst_entity] = []
                data['dst_' + src_entity + '_to_' + dst_entity] = []
                data['seq_' + src_entity + '_to_' + dst_entity] = []

            data['src_' + src_entity + '_to_' + dst_entity].append(src_num)
            data['dst_' + src_entity + '_to_' + dst_entity].append(dst_num)
            data['seq_' + src_entity + '_to_' + dst_entity].append(processed_neighbours[dst_node])

            processed_neighbours[dst_node] += 1  # this is useful to check which sequence number to use

        # this collects the sequence for the interleave aggregation (if any)
        for i in self.interleave_names:
            name, dst_entity = i
            interleave_definition = list(D_G.graph[name].values())  # this must be a graph variable

            involved_entities = {}
            total_sequence = []
            total_size, n_total, counter = 0, 0, 0

            for src_entity in interleave_definition:
                total_size += 1
                if src_entity not in involved_entities:
                    involved_entities[src_entity] = counter  # each entity a different value (identifier)

                    seq = data['seq_' + src_entity + '_to_' + dst_entity]
                    n_total += max(seq) + 1  # superior limit of the size of any destination
                    counter += 1

                # obtain all the original definition in a numeric format
                total_sequence.append(involved_entities[src_entity])

            # we exceed the maximum length for sake to make it multiple. Then we will cut it
            repetitions = math.ceil(float(n_total) / total_size)
            result = np.array((total_sequence * repetitions)[:n_total])

            for entity in involved_entities:
                id = involved_entities[entity]
                data['indices_' + entity + '_to_' + dst_entity] = np.where(result == id)[0].tolist()

        if self.training:
            return data, final_output
        else:
            return data

    def generate_from_array(self,
                            data_samples,
                            entity_names,
                            feature_names,
                            output_name,
                            interleave_names,
                            additional_input,
                            training,
                            shuffle=False):
        """
        Parameters
        ----------
        data_samples:    [array]
           Array of samples to be processed
        entity_names: [array]
            Name of the entities to be found in the dataset
        feature_names:    [array]
           Name of the features to be found in the dataset
        output_name:    str
           Name of the output data to be found in the dataset
        interleave_names:    [array]
           First parameter is the name of the interleave, and the second the destination entity
        additional_input:    [array]
           Name of other vectors that need to be retrieved because they appear in other parts of the model definition
        training:     bool
            Indicates if we are training, and thus a label is required.
        shuffle:    bool
           Shuffle parameter of the dataset
        """

        data_samples = [json.loads(x) for x in data_samples]
        self.entity_names = [x for x in entity_names]
        self.feature_names = [x for x in feature_names]
        self.output_name = output_name
        self.interleave_names = [[i[0], i[1]] for i in interleave_names]
        self.additional_input = [x for x in additional_input]
        self.training = training

        for sample in data_samples:
            try:
                processed_sample = self.__process_sample(sample)
                yield processed_sample

            except StopIteration:
                pass

            except KeyboardInterrupt:
                sys.exit

            except Exception as inf:
                print_failure("\n There was an unexpected error: \n" + str(inf))
                print_failure('Please make sure that all the names used in the sample passed ')

            sys.exit

    def generate_from_dataset(self,
                              dir,
                              entity_names,
                              feature_names,
                              output_name,
                              interleave_names,
                              additional_input,
                              training,
                              shuffle=False):
        """
        Parameters
        ----------
        dir:    str
           Path of the input dataset
        entity_names: [array]
            Name of the entities to be found in the dataset
        feature_names:    [array]
           Name of the features to be found in the dataset
        output_name:    str
           Name of the output data to be found in the dataset
        interleave_names:    [array]
           First parameter is the name of the interleave, and the second the destination entity
        additional_input:    [array]
           Name of other vectors that need to be retrieved because they appear in other parts of the model definition
        training:     bool
            Indicates if we are training, and thus a label is required.
        shuffle:    bool
           Shuffle parameter of the dataset
        """

        self.entity_names = entity_names
        self.feature_names = feature_names
        self.output_name = output_name
        self.interleave_names = interleave_names
        self.additional_input = additional_input
        self.training = training

        files = glob.glob(str(dir) + '/*.json') + glob.glob(str(dir) + '/*.tar.gz') + glob.glob(str(dir) + '/*.gml')
        # no elements found
        if files == []:
            raise Exception('The dataset located in  ' + dir + ' seems to contain no valid elements (json or .tar.gz)')

        if shuffle:
            random.shuffle(files)

        for sample_file in files:
            try:
                if 'tar.gz' in sample_file:
                    tar = tarfile.open(sample_file, 'r:gz')  # read the tar files
                    try:
                        file_name = tar.getmembers()[0]
                        file_samples = tar.extractfile(file_name)
                    except:
                        raise Exception('There was an error when trying to read the file ', file_name)
                else:
                    file_samples = open(sample_file, 'r')

                file_samples.read(1)
                data = self.stream_read_json(file_samples)

                while True:
                    processed_sample = self.__process_sample(next(data), sample_file)
                    yield processed_sample

            except StopIteration:
                pass

            except KeyboardInterrupt:
                sys.exit()

            except Exception as inf:
                print_info("\n There was an unexpected error: \n" + str(inf))
                print_info('Please make sure that all the names used in the file ' + sample_file +
                           ' are defined in your dataset')

                sys.exit()
