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
from ignnition.mp_classes import *
from functools import reduce
import importlib


class Yaml_preprocessing:
    """
    Class that handles all the information from a model_description file and organizes the information as a sequence of the corresponding auxiliary classes.

    Attributes
    ----------
    nn_architectures: [array]
    entities:   dict
        Contains the different entity object
    iterations_mp:  int
        Number of iterations that the mp phase will have
    mp_instances:   dict
        Contains the different MP object
    readout_op:    dict
        Information of the different combined message passings

    Methods:
    ----------
    __read_json(self, path)
        Reads the json file from the path and returns the data as a dictionary
    __read_yaml(self, path, name)
        Reads the yaml file from the path and returns the data as a dictionary
    __add_global_variables(self, data, global_variables)
        This method adds the values of the global variables to the data read from the model description file
    add_dimensions(self, dimensions)
        Computes the dimensions of the entities
    __validate_model_description(self, data)
        Validates the model description file by checking possible errors such as inexistent entity names, nn names or other possible misconfigurations
    __get_nn_mapping(self, models)
        Maps each of the NN by its name
    __get_entities(self, entities)
        Computes each of the entity objects, and adds to it its corresponding neural networks
    __add_nn_architectures_entities(self, entity)
        Adds the NN architecture corresponding to each of the NN references in the HS definition (by name)
    __add_nn_architecture_mp(self, m)
        Adds the NN architecture corresponding to each of the NN references in the rest of the parts of the model description file (by name)
    __get_mp_instances(self, inst)
        Computes the MP objects corresponding to the different message passings
    __add_readout_architecture(self, output)
        Adds the NN corresponding to the readout (wherever specified, by its referenced name)
    __get_readout_op(self, output_operations)
        Computes the pipeline of operations consistuiting the readout function
    get_input_dimensions(self)
        Returns the input dimension
    get_interleave_sources(self)
        Returns the source entities of a given interleave aggregation
    get_interleave_sources(self)
        Returns the number of iterations of the mp phase
    get_mp_iterations(self)
        Returns the number of iterations of the mp phase
    get_interleave_tensors(self)
        Returns a definition of the interleave pattern together with the destination name (entity name)
    get_readout_operations(self)
        Return the pipline of operations for the readout function
    get_output_info(self)
        Returns the operation of the output
    get_all_features(self)
        Returns all the feature names
    get_entity_names(self)
        Returns all the entity names
    get_adjacency_info(self)
        Returns the information of the adjacencies
    get_additional_input_names(self)
        Returns the names of any additional tensor that was referenced in the model_description and that doesn't fall in any of the previous categories
    """

    def __init__(self, model_dir):
        """
        Parameters
        ----------
        model_dir:    str
            Path of the yaml file with the model description
        """

        # read and validate the json file
        model_description_path = os.path.join(model_dir, 'model_description.yaml')
        self.data = self.__read_yaml(model_description_path, 'model description')

        # validate with the schema
        with importlib.resources.path('ignnition', "schema.json") as schema_file:
            validate(instance=self.data,schema=self.__read_json(schema_file))  # validate that the json is well defined

        # add the global variables (if any)
        global_variables_path = os.path.join(model_dir, 'global_variables.yaml')
        if os.path.exists(global_variables_path):
            global_variables = self.__read_yaml(global_variables_path, 'global variables')
            self.data = self.__add_global_variables(self.data, global_variables)

        else:
            print_info("No global_variables.yaml file detected in path: " + global_variables_path + ".\nIf you didn't use it in your model definition, ignore this message.")

        self.__validate_model_description(self.data)

        self.nn_architectures = self.__get_nn_mapping(self.data['neural_networks'])
        self.entities = self.__get_entities(self.data['entities'])

        self.iterations_mp = int(self.data['message_passing']['num_iterations'])
        self.mp_instances = self.__get_mp_instances(self.data['message_passing']['stages'])
        self.readout_op = self.__get_readout_op(self.data['readout'])

    # PRIVATE
    def __read_json(self, path):
        """
        Parameters
        ----------
        path:    str
            Path of the json file with the model description
        """

        with open(path) as json_file:
            return json.load(json_file)

    def __read_yaml(self, path, file_name=''):
        """
        Parameters
        ----------
        path:    str
            Path of the json file with the model description
        file_name: str
            Name of the file we aim to read
        """
        if os.path.isfile(path):
            with open(path, 'r') as stream:
                try:
                    return yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    print_failure("There was the following error in the " + file_name + " file.\n" + str(exc))
        else:
            print_failure("The " + file_name + " file was not found in: " + path)

    def __add_global_variables(self, data, global_variables):
        """
        Parameters
        ----------
        data:    dict
            Data of the model_description file
        global_variables: dict
            Dictionary mapping global variables to its corresponding names
        """
        if isinstance(data, dict):
            for k,v in data.items():
                if isinstance(v, str) and global_variables is not None and v in global_variables:
                        data[k] = global_variables[v]
                # make a recursive call
                elif isinstance(v, dict):
                    self.__add_global_variables(v, global_variables)
                elif isinstance(v, list):
                    for i in range(len(v)):
                        self.__add_global_variables(v[i], global_variables)
        return data

    # validate that all the nn_name are correct. Validate that all source and destination entities are correct. Validate that all the inputs in the message function are correct
    def __validate_model_description(self, data):
        """
        Parameters
        ----------
        data:    dict
           Dictionary with the initial data
        """

        stages = data['message_passing']['stages']

        src_names, dst_names, called_nn_names, input_names = [], [], [], []
        output_names = ['source', 'destination']

        for stage in stages:
            stage_mp = stage.get('stage_message_passings')
            for mp in stage_mp:  # for every message-passing
                dst_names.append(mp.get('destination_entity'))
                sources = mp.get('source_entities')

                # check the message functions
                for src in sources:
                    src_names.append(src.get('name'))
                    messages = src.get('message', None)
                    if messages is not None:
                        for op in messages:  # for every operation
                            if op.get('type') == 'neural_network':
                                called_nn_names.append(op.get('nn_name'))
                                input_names += op.get('input')

                            if 'output_name' in op:
                                output_names.append(op.get('output_name'))

                # check the aggregation functions
                aggregations = mp.get('aggregation')
                for aggr in aggregations:
                    if aggr.get('type') == 'neural_network':
                        input_names += aggr.get('input')

                    if 'output_name' in aggr:
                        output_names.append(aggr.get('output_name'))

        readout_op = data.get('readout')
        called_nn_names += [op.get('nn_name') for op in readout_op if op.get('type') == 'neural_network']

        if 'output_label' not in readout_op[-1]:
            print_failure('The last operation of the readout MUST contain the definition of the output_label')

        # now check the entities
        entity_names = [a.get('name') for a in data.get('entities')]
        nn_names = [n.get('nn_name') for n in data.get('neural_networks')]
        # check if the name of two NN defined match
        if len(nn_names) != len(set(nn_names)):
            print_failure("The names of two NN are repeated. Please ensure that each NN has a unique name.")


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

            # check the output and input names (ToDO: Change this since it may be referencing another feature from the dataset)
            #for i in input_names:
            #    if i not in output_names:
            #        raise Exception(
            #            'The name "' + i + '" was used as input of a message creation operation even though it was not the output of one.')

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

        return [Entity(self.__add_nn_architecture_entities(e)) for e in entities]

    def __add_nn_architecture_entities(self, entity):
        """
        Parameters
        ----------
        entity:    dict
           Add the information of the neural networks used in the entity
        """

        # add the message_creation nn architecture
        operations = entity.get('initial_state')
        for op in operations:
            if op.get('type') == 'neural_network':
                info = copy.deepcopy(self.nn_architectures[op['nn_name']])
                del op['nn_name']
                op['architecture'] = info.get('nn_architecture')
        return entity

    def __add_nn_architecture_mp(self, m):
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
                    if op.get('type') == 'neural_network':
                        info = copy.deepcopy(self.nn_architectures[op['nn_name']])
                        del op['nn_name']
                        op['architecture'] = info.get('nn_architecture')

        # add the aggregation nn architectures
        aggrs = m.get('aggregation')
        for aggr in aggrs:
            type =aggr.get('type')
            if type == 'edge_attention' or type == 'neural_network':
                info = copy.deepcopy(self.nn_architectures[aggr['nn_name']])
                del aggr['nn_name']
                aggr['architecture'] = info.get('nn_architecture')


        # add the update nn architecture
        if 'update' in m:
            if m['update']['type'] == 'neural_network':
                info = copy.deepcopy(self.nn_architectures[m['update']['nn_name']])
                del m['update']['nn_name']
                m['update']['architecture'] = info['nn_architecture']

        return m

    def __get_mp_instances(self, inst):
        """
        Parameters
        ----------
        inst:    dict
           Dictionary with the definition of each message passing
        """

        return [['stage_' + str(step_number), [Message_Passing(self.__add_nn_architecture_mp(m)) for m in stage['stage_message_passings']]] for step_number, stage in enumerate(inst)]

    def __add_readout_architecture(self, output):
        """
        Parameters
        ----------
        output:    dict
           Dictionary with the info of the readout architecture
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
            if type == 'pooling':
                result.append(Pooling_operation(op))

            elif type== 'product':
                result.append(Product_operation(op))

            elif type == 'neural_network':
                result.append(Feed_forward_operation(self.__add_readout_architecture(op), model_role = 'readout'))

            elif type == 'extend_adjacencies':
                result.append(Extend_adjacencies(op))

        return result

    # ----------------------------------------------------------------
    # PUBLIC FUNCTIONS
    def add_dimensions(self, dimensions):
        """
        Parameters
        ----------
        dimensions:      dict
            Dictionary mapping the dimension of each input
        """

        # add the dimensions of the entities
        for e in self.data['entities']:
            dimensions[e.get('name')] = e.get('state_dimension')
            dimensions[e.get('name') + '_initial'] = e.get('state_dimension')

        self.input_dim = dimensions # CHECK THIS
        del self.data

    # GETTERS
    def get_input_dimensions(self):
        return self.input_dim

    def get_interleave_sources(self):
        aux =  [[[src.name, mp.destination_entity] for src in mp.source_entities] for stage_name, mps in self.mp_instances for mp in mps if isinstance(mp.aggregations,Interleave_aggr)]
        return reduce(lambda accum, a: accum +a, aux, [])

    def get_mp_iterations(self):
        return self.iterations_mp

    def get_interleave_tensors(self):
        return [[mp.aggregations.combination_definition, mp.destination_entity] for stage_name, mps in self.mp_instances for mp in mps if isinstance(mp.aggregations, Interleave_aggr)]

    def get_mp_instances(self):
        return self.mp_instances

    def get_readout_operations(self):
        return self.readout_op

    def get_output_info(self):
        output_names = self.readout_op[-1].output_label
        return output_names

    def get_all_features(self):
        return reduce(lambda accum, e: accum + e.features_name, self.entities, [])

    def get_entity_names(self):
        return [e.name for e in self.entities]

    def get_adjacency_info(self):
        aux = [instance.get_instance_info() for step in self.mp_instances for instance in step[1]]
        return reduce(lambda accum, a: accum + a, aux, [])

    def get_additional_input_names(self):
        output_names = set()
        input_names = set()

        #gather here the inputs of the message operations (that are not source or destination)
        for stage in self.mp_instances:
            for mp in stage[1]:
                source_entities = mp.source_entities
                for s in source_entities:
                    for op in s.message_formation:  # for each operation
                        if op is not None and op.input is not None:
                            new_inputs = [i for i in op.input if i !='source' and i!= 'destination']
                            input_names.update(new_inputs)

                            if op.output_name is not None:
                                output_names.add(op.output_name)

        for r in self.readout_op:
            if r.type == 'extend_adjacencies':
                output_names.update(r.output_name)  # several names

            elif r.output_name is not None:
                output_names.add(r.output_name)

            for i in r.input:
                input_names.add(i)

        for e in self.entities:
            output_names.add(e.name)
            output_names.add(e.name + '_initial_state')

        return list(input_names.difference(output_names))

