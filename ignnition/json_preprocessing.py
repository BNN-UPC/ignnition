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


import yaml
from jsonschema import validate
import copy
import sys
import json
from ignnition.auxilary_classes import *
from functools import reduce
import importlib


class Json_preprocessing:
    """
    Attributes
    ----------
    entities:   dict
        Contains the different entity object

    iterations_mp:  int
        Number of iterations that the mp phase will have

    mp_instances:   dict
        Contains the different MP object

    comb_op:    dict
        Information of the different combined message passings

    outputs:    dict
        Readout architectures to be used for prediction

    training_op:   dict
        Training options and parameters

    entities_dimensions:    dict
        Maps each entity with its hidden state dimension.


    Methods:
    ----------
    __read_json(self, path)
        Reads the json file from the path and returns the data as a dictionary

    __obtain_entities(self, entities)
        Creates the entities specified in the entities dictionary

    __obtain_mp_instances(self, inst)
        Creates the message passing object

    __calculate_mp_combination_options(self, data)
        Creates the combined message passing object.

    __obtain_output_models(self, outputs)
        Creates the different readout models

    __obtain_training_options(self, data)
        Specifies the training options of the model

    __get_entities_dimensions(self)
        Obtains the dictionary that maps entities and its hidden-state dimensions

    get_entities_dimensions(self)
        Returns the dictionary that maps entities and its hidden-states dimensions

    get_entities(self)
        Returns the entities of the model

    get_combined_mp_options(self)
        Returns the combined message passings

    get_combined_mp_sources(self, dst_entity, step_name)
        Returns the sources that are part of a combined mp in step_name sending to dst_entity

    get_combined_sources(self)
        Return all sources that are part of a combined mp

    get_mp_iterations(self)
        Returns the number of iterations of the mp phase

    get_interleave_tensors(self)
        ?

    get_mp_instances(self)
        Returns the message passing objects of the model

    get_schedule(self)
        Returns the schedule set in the model

    get_optimizer(self)
        Returns the optimizer set in the model

    get_loss(self)
        Returns the loss set in the model

    get_output_operations(self)
        Returns the readout architectur of the model in the form of operations

    get_output_info(self)
        ?

    get_all_features(self)
        Returns all the features defined in the model, no matter the entity they are assigned to

    get_feature_size(self, feature)
        Returns the size of a feature

    get_adjecency_info(self)
    """

    def __init__(self, model_dir, dimensions, len1_features):
        """
        Parameters
        ----------
        path:    str (optional)
            Path of the json file with the model description
        """
        # read and validate the json file
        try:
            model_description_path = os.path.join(model_dir, 'model_description.yaml')
            data = self.__read_yaml(model_description_path)
        except:
            print_failure("The model_description.yaml file was not found in the path " + model_dir)

        # validate with the schema
        with importlib.resources.path('ignnition', "schema.json") as schema_file:
            validate(instance=data,schema=self.__read_json(schema_file))  # validate that the json is well defined

        # add the global variables
        try:
            global_variables_path = os.path.join(model_dir, 'global_variables.yaml')
            global_variables = self.__read_yaml(global_variables_path)
            data = self.__add_global_variables(data, global_variables)
        except:
            print_info("Global variables file not found")

        self.__validate_model_description(data)
        self.__add_dimensions(data, dimensions, len1_features)  # add the dimension of the features and of the edges
        self.nn_architectures = self.__get_nn_mapping(data['neural_networks'])
        self.entities = self.__get_entities(data['entities'])

        self.iterations_mp = int(data['message_passing']['num_iterations'])
        self.mp_instances = self.__get_mp_instances(data['message_passing']['stages'])
        self.readout_op = self.__get_readout_op(data['readout'])
        self.input_dim = self.__get_input_dims(dimensions)
        self.weight_matrices = self.__get_weight_matrices(data['neural_networks'])

    # PRIVATE
    def __read_json(self, path):
        """
        Parameters
        ----------
        path:    str (optional)
            Path of the json file with the model description
        """
        with open(path) as json_file:
            return json.load(json_file)

    def __read_yaml(self, path):
        """
        Parameters
        ----------
        path:    str (optional)
            Path of the json file with the model description
        """
        with open(path, 'r') as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print("The model description file was not found in: " + path)

    def __add_global_variables(self, data, global_variables):
        if isinstance(data, dict):
            for k,v in data.items():
                if isinstance(v, str) and v in global_variables:
                    data[k] = global_variables[v]

                # make a recursive call
                elif isinstance(v, dict):
                    self.__add_global_variables(v, global_variables)
                elif isinstance(v, list):
                    for i in range(len(v)):
                        self.__add_global_variables(v[i], global_variables)

        return data

    def __add_dimensions(self, data, dimensions, len1_features):
        """
        Parameters
        ----------
        data:    dict
            Dictionary with the initial data
        dimensions:      dict
            Dictionary mapping the dimension of each input
        """

        for e in data['entities']:
            new_features = []
            features = e.get('features')    # names of the features
            for name in features:
                if name in len1_features:
                    size = 1
                else:
                    size = dimensions.get(name)
                new_features.append({"name": name, "size": size})
            e['features'] = new_features

        for stage in data['message_passing']['stages']:
            stages = stage.get('stage_message_passings')
            for mp in stages:
                sources = mp.get('source_entities')
                for src in sources:
                    src['extra_parameters'] = dimensions.get(src['adj_vector'])

    # validate that all the nn_name are correct. Validate that all source and destination entities are correct. Validate that all the inputs in the message function are correct
    def __validate_model_description(self, data):
        """
        Parameters
        ----------
        data:    dict
           Dictonary with the initial data
        """
        stages = data['message_passing']['stages']

        src_names, dst_names, called_nn_names, input_names = [], [], [], []
        output_names = ['hs_source', 'hs_dst', 'edge_params']

        for stage in stages:
            stage_mp = stage.get('stage_message_passings')
            for mp in stage_mp:  # for every message-passing
                dst_names.append(mp.get('destination_entity'))
                sources = mp.get('source_entities')
                for src in sources:
                    src_names.append(src.get('name'))
                    messages = src.get('message', None)
                    if messages is not None:
                        for op in messages:  # for every operation
                            if op.get('type') == 'feed_forward':
                                called_nn_names.append(op.get('nn_name'))
                                input_names += op.get('input')

                            if 'output_name' in op:
                                output_names.append(op.get('output_name'))

        readout_op = data.get('readout')
        called_nn_names += [op.get('nn_name') for op in readout_op if op.get('type') == 'feed_forward']

        if 'output_label' not in readout_op[-1]:
            print_failure('The last operation of the readout MUST contain the definition of the output_label')

        # now check the entities
        entity_names = [a.get('name') for a in data.get('entities')]
        nn_names = [n.get('nn_name') for n in data.get('neural_networks')]

        try:
            # check the source entities
            for a in src_names:
                if a not in entity_names:
                    raise Exception(
                        'The source entity "' + a + '" was used in a message passing. However, there is no such entity. \n Please check the spelling or define a new entity.')

            # check the destination entities
            for d in dst_names:
                if d not in entity_names:
                    raise Exception(
                        'The destination entity "' + d + '" was used in a message passing. However, there is no such entity. \n Please check the spelling or define a new entity.')

            # check the nn_names
            for name in called_nn_names:
                if name not in nn_names:
                    raise Exception(
                        'The name "' + name + '" is used as a reference to a neural network (nn_name), even though the neural network was not defined. \n Please make sure the name is correctly spelled or define a neural network named ' + name)

            # check the output and input names
            for i in input_names:
                if i not in output_names:
                    raise Exception(
                        'The name "' + i + '" was used as input of a message creation operation even though it was not the output of one.')

        except Exception as inf:
            print_failure(str(inf) + '\n')

    def __get_nn_mapping(self, models):
        """
        Parameters
        ----------
        models:    array
           Array of nn models
        """

        result = {}
        for m in models:
            result[m.get('nn_name')] = m
        return result

    def __get_entities(self, entities):
        """
        Parameters
        ----------
        entities:    dict
           Dictionary with the definition of each entity
        """
        return [Entity(e) for e in entities]


    def __add_nn_architecture(self, m):
        """
        Parameters
        ----------
        m:    dict
           Add the information of the nn architecture
        """

        # add the message_creation nn architecture
        sources = m.get('source_entities')
        for s in sources:
            messages = s.get('message', None)
            if messages is not None:
                for op in messages:
                    if op.get('type') == 'feed_forward':
                        info = copy.deepcopy(self.nn_architectures[op['nn_name']])
                        del op['nn_name']
                        op['architecture'] = info.get('nn_architecture')

        aggr = m['aggregation']
        if aggr.get('type') == 'edge_attention':
            info = copy.deepcopy(self.nn_architectures[aggr['nn_name']])
            del aggr['nn_name']
            aggr['architecture'] = info.get('nn_architecture')

        # add the update nn architecture
        if 'update' in m:
            if m['update']['type'] == 'feed_forward':
                info = copy.deepcopy(self.nn_architectures[m['update']['nn_name']])
                del m['update']['nn_name']
                m['update']['architecture'] = info['nn_architecture']

            if m['update']['type'] == 'recurrent_neural_network':
                architecture = copy.deepcopy((self.nn_architectures[m['update']['nn_name']]))
                del m['update']['nn_name']

                for k, v in architecture.items():
                    if k != 'nn_name' and k != 'nn_type':
                        m['update'][k] = v
        return m

    def __get_mp_instances(self, inst):
        """
        Parameters
        ----------
        inst:    dict
           Dictionary with the definition of each message passing
        """
        return [['stage_' + str(step_number), [Message_Passing(self.__add_nn_architecture(m)) for m in stage['stage_message_passings']]] for step_number, stage in enumerate(inst)]

    def __add_readout_architecture(self, output):
        """
        Parameters
        ----------
        output:    dict
           Dictonary with the info of the readout architecture
        """

        name = output['nn_name']
        info = copy.deepcopy(self.nn_architectures.get(name))
        del output['nn_name']
        output['architecture'] = info['nn_architecture']

        return output

    def __get_readout_op(self, output_operations):
        """
        Parameters
        ----------
        output_operations:    dict
           List of dictionaries with the definition of the operations forming one readout model
        """
        result = []
        for op in output_operations:
            type = op.get('type')
            if type == 'predict':
                result.append(Predicting_operation(op))

            elif type == 'pooling':
                result.append(Pooling_operation(op))

            elif type== 'product':
                result.append(Product_operation(op))

            elif type == 'feed_forward':
                result.append(Feed_forward_operation(self.__add_readout_architecture(op), model_role = 'readout'))

            elif type == 'extend_adjacencies':
                result.append(Extend_adjacencies(op))

        return result

    def __get_input_dims(self, dimensions):
        """
        Parameters
        ----------
        dimensions:    dict
           Dictionary with the dimensions of the input
        """

        dict = {}
        for entity in self.entities:
            dict[entity.name] = entity.hidden_state_dimension
            dict[entity.name + '_initial'] = entity.hidden_state_dimension

        # add the size of additional inputs if needed
        return {**dict, **dimensions}


    def __get_weight_matrices(self, neural_networks):
        return [Weight_matrix(nn) for nn in neural_networks if nn.get('nn_type') == 'weight_matrix']



    # ----------------------------------------------------------------
    # PUBLIC FUNCTIONS GETTERS
    def get_input_dimensions(self):
        return self.input_dim

    def get_entities(self):
        return self.entities

    def get_combined_mp_sources(self, dst_entity, step_name):
        """
        Parameters
        ----------
        dst_entity:    str
           Name of the destination entity
        step_name:      str
            Name of the message passing step
        """

        sources = []
        for step in self.mp_instances:
            if step[0] == step_name:  # check if we are in fact within the step we care about
                for m in step[1]:  # this is just one value
                    if (m.type == "multi_source") & (m.destination_entity == dst_entity):
                        sources.append(m.source_entity)

        return sources

    def get_interleave_sources(self):
        aux =  [[[src.name, mp.destination_entity] for src in mp.source_entities] for stage_name, mps in self.mp_instances for mp in mps if isinstance(mp.aggregation,Interleave_aggr)]
        return reduce(lambda accum, a: accum +a, aux, [])

    def get_mp_iterations(self):
        return self.iterations_mp

    def get_interleave_tensors(self):
        return [[mp.aggregation.combination_definition, mp.destination_entity] for stage_name, mps in self.mp_instances for mp in mps if isinstance(mp.aggregation, Interleave_aggr)]

    def get_mp_instances(self):
        return self.mp_instances

    def get_readout_operations(self):
        return self.readout_op

    def get_output_info(self):
        output_names = self.readout_op[-1].output_label
        return output_names

    def get_all_features(self):
        return reduce(lambda accum, e: accum + e.features, self.entities, [])

    def get_feature_size(self, feature):
        """
        Parameters
        ----------
        feature:    dict
           Dictionary with the sizes of each feature
        """
        return feature['size'] if 'size' in feature else 1

    def get_entity_names(self):
        return [e.name for e in self.entities]

    def get_adjecency_info(self):
        aux = [instance.get_instance_info() for step in self.mp_instances for instance in step[1]]
        return reduce(lambda accum, a: accum + a, aux, [])

    def get_additional_input_names(self):
        output_names = set()
        input_names = set()

        for r in self.readout_op:
            if r.type == 'extend_adjacencies':
                output_names.update(r.output_name)  # several names

            elif r.output_name is not None:
                output_names.add(r.output_name)

            for i in r.input:
                input_names.add(i)

        for e in self.entities:
            output_names.add(e.name)
            output_names.add(e.name + '_initial')

        return list(input_names.difference(output_names))


    def get_weight_matrices(self):
        return self.weight_matrices