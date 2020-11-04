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


class Generator:
    def __init__(self):
        self.end_symbol = bytes(']', 'utf-8')

    def __cummax(self, alist, extractor):
        with tf.name_scope('cummax'):
            maxes = [tf.math.reduce_max(extractor(v)) + 1 for v in alist]
            cummaxes = [tf.zeros_like(maxes[0])]
            for i in range(len(maxes) - 1):
                cummaxes.append(tf.math.add_n(maxes[0:i + 1]))

        return cummaxes


    def make_indices(self, sample, entity_names):
        """
        Parameters
        ----------
        entities:    dict
           Dictionary with the information of each entity
        """
        counter = {}
        indices = {}
        for e in entity_names:
            items = sample[e]
            indices[e] = {}
            counter[e] = 0
            for node in items:
                indices[e][node] = counter[e]
                counter[e] += 1

        return counter, indices

    def stream_read_json(self, f):
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
                start_pos += e.pos +1
                a = f.read(1)   # this 1 is the coma or the final symbol

                if a == self.end_symbol or a == ']':
                    yield obj
                    return
                yield obj

    def __process_sample(self, sample):
        data = {}
        output = []
        # check that all the entities are defined
        for e in self.entity_names:
            if e not in sample:
                raise Exception('The entity "' + e + '" was used in the model_description.json file '
                                                     'but was not defined in the dataset. A list should be defined with the names (string) of each node of type ' + e + '.\n'
                                                                                                                                                                        'E.g., "' + e + '": ["n1", "n2", "n3" ...]')

        # read the features
        for f in self.feature_names:
            if f not in sample:
                raise Exception(
                    'IGNNITION: The feature "' + f + '" was used in the model_description.json file '
                                                     'but was not defined in the dataset. A list should be defined with the corresponding value'
                                                     'for each of the nodes of its entity type. \n'
                                                     'E.g., "' + f + '": [v1, v2, v3 ...]')
            else:
                data[f] = sample[f]

        # read additional input name
        for a in self.additional_input:
            if a not in sample:
                raise Exception('There was an list of float values named"' + str(
                    a) + '" which was used in the model_description.json file. Thus it must be defined in the dataset. Please make sure the spelling is correct.')
            else:
                data[a] = sample[a]

        # read the output values if we are training
        if self.training:
            if self.output_name not in sample:
                raise Exception('The model_description.json file defined a label named "' + str(
                    self.output_name) + '". This was, however, not found. Make sure the spelling is correct, and that it is a valid array of float values.')
            else:
                value = sample[self.output_name]
                if not isinstance(value, list):
                    value = [value]

                output += value

        dict = {}

        num_nodes, indices = self.make_indices(sample, self.entity_names)

        # create the adjacencies
        for a in self.adj_names:
            name, src_entity, dst_entity, uses_parameters = a

            if name not in sample:
                raise Exception(
                    'A list for the adjecency list named "' + name + '" was not found although being expected.\n'
                                                                     'Remember that an adjacecy list connecting entity "a" to "b" should be defined as follows:\n'
                                                                     '{"b1":["a1","a2",...], "b2":["a2","a4"...]')

            else:
                adjecency_lists = sample[name]
                src_idx, dst_idx, seq, parameters = [], [], [], []

                # ordered always by destination. (p1: [l1,l2,l3], p2:[l4,l5,l6]...
                items = adjecency_lists.items()

                for destination, sources in items:
                    try:
                        indices_src = indices[src_entity]
                    except:
                        raise Exception(
                            'The adjecency list "' + name + '" was expected to be from ' + src_entity + ' to ' + dst_entity +
                            '.\n However the source entity defined in the dataset does not match')

                    try:
                        indices_dst = indices[dst_entity]
                    except:
                        raise Exception(
                            'The adjecency list "' + name + '" was expected to be from ' + src_entity + ' to ' + dst_entity +
                            '.\n However the destination entity defined in the dataset does not match')

                    seq += range(0, len(sources))

                    # check if this adjacency contains extra parameters. This would mean that the sources array would be of shape p0:[[l0,params],[l1,params]...]
                    if isinstance(sources[0], list):
                        for s in sources:
                            src_name = s[0]
                            src_idx.append(indices_src[src_name])
                            dst_idx.append(indices_dst[destination])

                            # add the parameters. s[1] should indicate its name
                            if uses_parameters == 'True':   parameters.append(s[1])

                    # in case no extra parameters are provided
                    else:
                        for s in sources:
                            src_idx.append(indices_src[s])
                            dst_idx.append(indices_dst[destination])

                data['src_' + name] = src_idx
                data['dst_' + name] = dst_idx

                # add sequence information
                data['seq_' + src_entity + '_' + dst_entity] = seq
                dict['seq_' + src_entity + '_' + dst_entity] = seq

                # remains to check that all adjacencies of the same type have params or not (not a few of them)!!!!!!!!!!
                if parameters != []:
                    data['params_' + name] = parameters

        # define the graph nodes
        items = num_nodes.items()
        for entity, n_nodes in items:
            data['num_' + entity] = n_nodes

        # obtain the sequence for the combined message passing. One for each source entity sending to the destination.
        for i in self.interleave_names:
            name, dst_entity = i
            interleave_definition = sample[name]

            involved_entities = {}
            total_sequence = []
            total_size, n_total, counter = 0, 0, 0

            for entity in interleave_definition:
                total_size += 1
                if entity not in involved_entities:
                    involved_entities[entity] = counter  # each entity a different value (identifier)

                    seq = dict['seq_' + entity + '_' + dst_entity]
                    n_total += max(seq) + 1  # superior limit of the size of any destination
                    counter += 1

                # obtain all the original definition in a numeric format
                total_sequence.append(involved_entities[entity])

            # we exceed the maximum length for sake to make it multiple. Then we will cut it
            repetitions = math.ceil(float(n_total) / total_size)
            result = np.array((total_sequence * repetitions)[:n_total])

            for entity in involved_entities:
                id = involved_entities[entity]
                data['indices_' + entity + '_to_' + dst_entity] = np.where(result == id)[0].tolist()

        if self.training:
            return data, output
        else:
            return data

    def __get_new_item(self, data, i):
        # if the data was passed in the form of an array
        if isinstance(data, list):
            return data[i]
        return next(data)

    def __process_minibatch(self, data, batch_size):
        data_features = {}
        data_target = []
        for i in range(batch_size):
            processed_sample = self.__process_sample(self.__get_new_item(data, i))

            if not isinstance(processed_sample, tuple):
                processed_sample = [processed_sample]

            if i == 0:
                data_features = processed_sample[0]
                if self.training:
                    data_target = processed_sample[1]

            else:
                # cummax all the adjacency list (both src, dst and seq)
                for a in self.adj_names:
                    max_value = max(data_features['src_' + a[0]])

                    data_features['src_' + a[0]] += (np.array(processed_sample[0]['src_' + a[0]]) + max_value).tolist()

                    max_value = max(data_features['dst_' + a[0]])
                    data_features['dst_' + a[0]] += (np.array(processed_sample[0]['dst_' + a[0]]) + max_value).tolist()

                    data_features['seq_' + a[1] + '_' + a[2]] += processed_sample[0]['seq_' + a[1] + '_' + a[2]]

                for f in self.feature_names:
                    data_features[f] += processed_sample[0][f]

                for a in self.additional_input:
                    data_features[a] += processed_sample[0][a]

                for e in self.entity_names:#this is useful to approach this problem more generally for both training and predictions
                    data_features['num_' + e] += processed_sample[0]['num_' + e]

                for i in self.interleave_names:
                    max_value = max(data_features['indices_' + i[0] + '_to_' + i[1]])
                    data_features['indices_' + i[0] + '_to_' + i[1]] += (
                                np.array(processed_sample[0]['indices_' + i[0] + '_to_' + i[1]]) + max_value).tolist()

                # process the target
                if self.training:
                    data_target += processed_sample[1]

        if self.training:
            return data_features, data_target

        return data_features


    def generate_from_array(self,
                            data_samples,
                            entity_names,
                            feature_names,
                            output_name,
                            adj_names,
                            interleave_names,
                            additional_input,
                            training,
                            shuffle=False,
                            batch_size= 8):
        """
        Parameters
        ----------
        dir:    str
           Name of the entity
        feature_names:    str
           Name of the features to be found in the dataset
        output_names:    str
           Name of the output data to be found in the dataset
        adj_names:    [array]
           CHECK
        interleave_names:    [array]
           First parameter is the name of the interleave, and the second the destination entity
        predict:     bool
            Indicates if we are making predictions, and thus no label is required.
        shuffle:    bool
           Shuffle parameter of the dataset

        """
        data_samples = [json.loads(x.decode('ascii')) for x in data_samples]
        self.entity_names = [x.decode('ascii') for x in entity_names]
        self.feature_names = [x.decode('ascii') for x in feature_names]
        self.output_name = output_name.decode('ascii')
        self.adj_names = [[x[0].decode('ascii'), x[1].decode('ascii'), x[2].decode('ascii'), x[3].decode('ascii')] for x
                          in
                          adj_names]
        self.interleave_names = [[i[0].decode('ascii'), i[1].decode('ascii')] for i in interleave_names]
        self.additional_input = [x.decode('ascii') for x in additional_input]
        self.training = training
        samples = glob.glob(str(dir) + '/*.tar.gz')

        if shuffle == True:
            random.shuffle(samples)

        n = len(data_samples)
        for i in n:
            try:
                mini_batch_samples = data_samples[i:i+batch_size]
                mini_batch = self.__process_minibatch(mini_batch_samples, batch_size)
                yield mini_batch

            except StopIteration:
                pass

            except KeyboardInterrupt:
                sys.exit

            except Exception as inf:
                print_info("There was an unexpected error: \n" +  str(inf))
                print_info('Please make sure that all the names used for the definition of the model '
                    'are defined in your dataset. For instance, you should define a list for: \n'
                    '1) A list for each of the entities defined with all its nodes of the graph\n'
                    '2) Each of the features used to define an entity\n'
                    '3) Additional lists/values used for the definition\n'
                    '4) The label aimed to predict\n'
                    '---------------------------------------------------------')
                sys.exit


    def generate_from_dataset(self,
                              dir,
                              entity_names,
                              feature_names,
                              output_name,
                              adj_names,
                              interleave_names,
                              additional_input,
                              training,
                              shuffle=False,
                              batch_size = 8):
        """
        Parameters
        ----------
        dir:    str
           Name of the entity
        feature_names:    str
           Name of the features to be found in the dataset
        output_names:    str
           Name of the output data to be found in the dataset
        adj_names:    [array]
           CHECK
        interleave_names:    [array]
           First parameter is the name of the interleave, and the second the destination entity
        predict:     bool
            Indicates if we are making predictions, and thus no label is required.
        shuffle:    bool
           Shuffle parameter of the dataset

        """
        self.entity_names = entity_names
        self.feature_names = feature_names
        self.output_name = output_name
        self.adj_names = adj_names
        self.interleave_names = interleave_names
        self.additional_input = additional_input
        self.training = training

        samples =  glob.glob(str(dir) + '/*.json') + glob.glob(str(dir) + '/*.tar.gz')
        if shuffle:
            random.shuffle(samples)

        for sample_file in samples:
            try:
                if 'tar.gz' in sample_file:
                    tar = tarfile.open(sample_file, 'r:gz')  # read the tar files
                    try:
                        file_samples = tar.extractfile('data.json')
                    except:
                        raise Exception('The file data.json was not found in ', sample_file)

                else:
                    file_samples = open(sample_file, 'r')

                file_samples.read(1)
                aux = self.stream_read_json(file_samples)
                while True:
                    mini_batch = self.__process_minibatch(aux, batch_size)
                    yield mini_batch

            except StopIteration:
                pass

            except KeyboardInterrupt:
                sys.exit

            except Exception as inf:
                print_info("\n There was an unexpected error: \n" + str(inf))
                print_info('Please make sure that all the names used for the definition of the model '
                              'are defined in your dataset. For instance, you should define a list for: \n'
                              '1) A list for each of the entities defined with all its nodes of the graph\n'
                              '2) Each of the features used to define an entity\n'
                              '3) Additional lists/values used for the definition\n'
                              '4) The label aimed to predict\n'
                              '---------------------------------------------------------')
                sys.exit
