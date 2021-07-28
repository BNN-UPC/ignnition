"""
 *
 * Copyright (C) 2020 Universitat Politècnica de Catalunya.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
"""

# -*- coding: utf-8 -*-

import sys
import glob
import tarfile
import math
import random
import functools
import json

import numpy as np
import networkx as nx

from ignnition.utils import print_failure, print_info

from networkx.readwrite import json_graph


class Generator:
    """
    This class implements the Generator in charge of feeding the data to the main GNN module. This class will take as
    input the original datasets of the user or the passed array and compute a series of transformation and
    precalculations. Finally it serves it to the GNN module.

    Attributes
    ----------
    end_symbol:    str
        End symbol to be used when reading the json file as a stream of data.

    Methods:
    ----------
    stream_read_json(self, f)
       Creates a generator of samples from the given dataset or sample array. All these are read as a stream of data
       to avoid full alocation on memory.

    __process_sample(self, sample, file=None)
        Given an input sample, it processes it and pre-computes several aspects to be later served to the GNN module.

    generate_from_array
        Creates and returns the generator from an input array of samples of the user.

    generate_from_dataset
        Creates and returns the generator from an input dataset of samples of the user.
    """

    def __init__(self):
        self.end_symbol = bytes(']', 'utf-8')
        self.warnings_shown = False

    def stream_read_json(self, f):
        """
        Parameters
        ----------
        f:
            Input data
        """
        # check that it is a valid array of objects
        pos1 = f.read(1)
        if pos1 != '[':
            print_failure(
                "Error because the dataset files must be an array of json objects, and not single json objects")

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

        # Only directed graphs are supported. Error checking message if the graph is undirected
        if not G.is_directed():
            print_failure("IGNNITION received as input an undirected graph, even though it only supports "
                          "(at the moment) directed graphs. Please consider reformating your code accordingly. "
                          "You can double the edges between two nodes (e.g., edge 1-2 can be transformed into 1->2 "
                          "and 2->1) to simulate the same behaviour.")

        if G.is_multigraph():
            print_failure("IGNNITION received as input a multigraph, while these are not yet supported. This means, "
                          "that for every pair of nodes, only one edge with the same source and destination can exist "
                          "(e.g., you cannot have two edges 1->2 and 1->2. Notice that 1->2 and 2->1 does not incur "
                          "in this problem.")

        entity_counter = {}
        mapping = {}
        data = {}

        for name in self.entity_names:
            entity_counter[name] = 0

        list_nodes = list(G.nodes())
        for node_name in list_nodes:
            attributes = G.nodes[node_name]

            if 'entity' not in attributes:
                print_failure(
                    "Error in the dataset file located in '" + file + ". The node named'" + str(node_name)
                    + "' was not assigned an entity.")

            entity_name = attributes['entity']
            new_node_name = entity_name + '_{}'
            num_node = entity_counter[entity_name]
            entity_counter[entity_name] += 1

            mapping[node_name] = new_node_name.format(num_node)

        # save the number of nodes of each entity
        for name in self.entity_names:
            data['num_' + name] = entity_counter[name]

        # rename the name of the nodes to a mapping that also indicates its entity layer_type
        D_G = nx.relabel_nodes(G, mapping)

        # discard if the graph is empty
        if not D_G.edges():
            print_info("\nA sample was discarded because the graph is empty (has no edges).")
            raise StopIteration

        # load the features (all the features are set to be lists. So we always return a list of lists)
        for f in self.feature_names:
            try:
                features_dict = nx.get_node_attributes(D_G, f)
                feature_vals = np.array(list(features_dict.values()))
                entity_names = set([name.split('_')[0] for name in features_dict.keys()])   #indicates the (unique)
                # names of the entities that have that feature

                if len(entity_names) > 1:
                    entities_string = functools.reduce(lambda x,y: str(x) + ',' + str(y), entity_names )
                    print_failure("The feature " + f + " was defined in several entities(" + entities_string +
                                  "). The feature names should be unique for each layer_type of node.")

                # it should always be a 2d array
                if len(np.shape(feature_vals)) == 1:
                    feature_vals = np.expand_dims(feature_vals, axis=-1)

                if feature_vals.size == 0:
                    message = "The feature " + f + " was used in the model_description.yaml file " \
                                                   "but was not defined in the dataset."
                    if file is not None:
                        message = "Error in the dataset file located in '" + file + ".\n" + message
                    raise Exception(message)
                else:
                    data[f] = feature_vals

            except:
                message = "The feature " + f + " was used in the model_description.yaml file " \
                                               "but was not defined in the dataset."
                if file is not None:
                    message = "Error in the dataset file located in '" + file + ".\n" + message
                raise Exception(message)

        # take other inputs if needed (check that they might be global features)
        for a in self.additional_input:
            # 1) try to see if this name has been defined as a node attribute
            node_dict = nx.get_node_attributes(D_G, a)
            node_attr = np.array(list(node_dict.values()))
            entity_names = set([name.split('_')[0] for name in node_dict.keys()])  # indicates the (unique) names
            # of the entities that have that feature

            if len(entity_names) > 1:
                entities_string = functools.reduce(lambda x, y: str(x) + ',' + str(y), entity_names)
                print_failure(
                    "The feature " + a + " was defined in several entities(" + entities_string +
                    "). The feature names should be unique for each layer_type of node.")

            # it should always be a 2d array
            if len(np.shape(node_attr)) == 1:
                node_attr = np.expand_dims(node_attr, axis=-1)

            # 2) try to see if this name has been defined as an edge feature
            edge_dict = nx.get_edge_attributes(D_G, a)
            edge_attr = np.array(list(edge_dict.values()))
            entity_names = set([(pair[0].split('_')[0], pair[1].split('_')[0]) for pair in edge_dict.keys()])  #
            # indicates the (unique) names of the entities that have that feature
            # obtain the entities, with a small token indicating if it is source or destination

            # Problem: When we transform an undirected graph to directed, we double all the edges. Hence,
            # we still need to differentiate between source and destination entities?? Solution: Allow only directed??

            # for now, check that the name is unique for every src-dst. Problem: One node connected to another but
            # the reverse to other nodes??
            if len(entity_names) > 2:
                #print(entity_names)
                entities_string = functools.reduce(lambda x, y: str(x) + ',' + str(y), entity_names)
                print_failure(
                    "The edge feature " + a + " was defined in connecting two different source-destination entities(" +
                    entities_string + "). Make sure that an edge feature is unique for a given pair of entities "
                                      "(types of nodes).")


            # it should always be a 2d array
            if len(np.shape(edge_attr)) == 1:
                edge_attr = np.expand_dims(edge_attr, axis=-1)


            # 3) try to see if this name has been defined as a graph feature
            graph_attr = [D_G.graph[a]] if a in D_G.graph else []

            # Check that this name has not been defined both as node features and as edge_features
            if node_attr.size != 0 and edge_attr.size != 0 and len(graph_attr) != 0:
                print_failure("The feature " + a + "was defined both as node feature, edge feature and graph feature. "
                                                   "Please use unique names in this case.")
            elif node_attr.size != 0 and edge_attr.size != 0:
                print_failure("The feature " + a + "was defined both as node feature and as edge feature. Please use "
                                                   "unique names in this case.")
            elif node_attr.size != 0 and len(graph_attr) != 0:
                print_failure("The feature " + a + "was defined both as node feature and as graph feature. Please use "
                                                   "unique names in this case.")


            # Return the correct value
            if node_attr.size != 0:
                data[a] = node_attr
            elif edge_attr.size != 0:
                data[a] = edge_attr
            elif a in D_G.graph:
                data[a] = graph_attr
            else:
                message = 'The data named "' + a + '" was used in the model_description.yaml file ' \
                                                   'but was not defined in the dataset.'
                if file is not None:
                    message = "Error in the dataset file located in '" + file + ".\n" + message
                raise Exception(message)

        if self.training:
            # collect the output (if there is more than one, concatenate them on axis=1
            # limitation: all the outputs must be of the same layer_type (same number of elements)
            final_output = []
            for output in self.output_names:
                try:
                    aux = list(nx.get_node_attributes(D_G, output).values())
                    if not aux:  # When having global/graph-level output
                        aux = D_G.graph[output]
                        aux = aux if isinstance(aux, list) else [aux]

                except Exception:
                    print_failure(
                        f"Error when trying to get output with name: {output}. "
                        "Check the data which corresponds to the output_label in the readout block."
                    )

                # if it is a 1d array, transform it into a 2d array
                if len(np.array(aux).shape) == 1:
                    aux = np.expand_dims(aux, -1)

                # try to concatenate them together. If error, it means that the two labels are incompatible
                final_output.extend(aux)
                data['__ignnition_{}_len'.format(output)] = len(aux)

        # find the adjacencies
        edges_list = list(D_G.edges())
        processed_neighbours = {}

        # create the adjacency lists that we are required to pass
        for adj_name_item in self.adj_names:
            src_entity = adj_name_item.split('_to_')[0]
            dst_entity = adj_name_item.split('_to_')[1]

            data['src_' + src_entity + '_to_' + dst_entity] = []
            data['dst_' + src_entity + '_to_' + dst_entity] = []
            data['seq_' + src_entity + '_to_' + dst_entity] = []

        for e in edges_list:
            src_node, dst_node = e
            src_num = int(src_node.split('_')[-1])
            dst_num = int(dst_node.split('_')[-1])
            src_entity = D_G.nodes[src_node]['entity']
            dst_entity = D_G.nodes[dst_node]['entity']
            if dst_node not in processed_neighbours:
                processed_neighbours[dst_node] = 0

            if src_entity + '_to_' + dst_entity in self.adj_names:
                data['src_' + src_entity + '_to_' + dst_entity].append(src_num)
                data['dst_' + src_entity + '_to_' + dst_entity].append(dst_num)
                data['seq_' + src_entity + '_to_' + dst_entity].append(processed_neighbours[dst_node])

                processed_neighbours[dst_node] += 1  # this is useful to check which sequence number to use


        # check that the dataset contains all the adjacencies needed
        if not self.warnings_shown:
            for adj_name_item in self.adj_names:
                if data['src_' + adj_name_item] == []:
                    src_entity = adj_name_item.split('_to_')[0]
                    dst_entity = adj_name_item.split('_to_')[1]
                    print_info(
                        "WARNING: The GNN definition uses edges between " + src_entity + " and " + dst_entity +
                        " but these were not found in the input graph. The MP defined between these two entities "
                        "will be ignored.\nIn case the graph ought to contain such edges, one reason for this error "
                        "is a mistake in defining the graph as directional, when the edges have been defined as "
                        "undirected. Please check the documentation.")
                    self.warnings_shown = True

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
                            output_names,
                            adj_names,
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
        output_names:    [str]
           Names of the output data to be found in the dataset
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
        self.output_names = output_names
        self.adj_names = adj_names
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
                sys.exit()

            except Exception as inf:
                print_failure("\n There was an unexpected error: \n" + str(
                    inf) + "\n Please make sure that all the names used in the sample passed ")

    def generate_from_dataset(self,
                              dir,
                              entity_names,
                              feature_names,
                              output_names,
                              adj_names,
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
        output_names:    [str]
           Name of the output data to be found in the dataset
        interleave_names:    [array]
           First parameter is the name of the interleave, and the second the destination entity
        additional_input:    [array]
           Name of other vectors that need to be retrieved because they appear in other parts of the model definition
        training:     bool
            Indicates if we are training, and thus a label is required.
        shuffle:    bool
           Shuffle parameter of the dataset

        Args:
            adj_names:
        """

        self.entity_names = entity_names
        self.feature_names = feature_names
        self.output_names = output_names
        self.adj_names = adj_names
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

                data = self.stream_read_json(file_samples)

                while True:
                    processed_sample = self.__process_sample(next(data), sample_file)
                    yield processed_sample

            except StopIteration:
                pass

            except KeyboardInterrupt:
                sys.exit()

            #except Exception as inf:
            #    print_failure("\n There was an unexpected error: \n" + str(
            #        inf) + "\n Please make sure that all the names used in the file: " + sample_file +
            #                  ' are defined in your dataset')
