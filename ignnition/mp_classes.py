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


import tensorflow as tf
import tensorflow.keras.activations
from keras import backend as K
import sys
from ignnition.utils import *
from ignnition.operation_classes import *
from ignnition.aggregation_classes import *


class Entity:
    """
    A class that represents an entity of the MSMP graph

    Attributes
    ----------
    name:    str
        Name of the feature
    state_dimension:   int
        Dimension of the hiddens state of these entity node's
    operations:  array
        Array with all the operations to construct the hidden_states

    Methods:
    ----------
    compute_hs_operations(self, operations)
        Returns the corresponding hs of this entity (tensorflow object) by performing the set of operations passed by parameters

    get_entity_total_feature_size(self)
        Sum of the dimension of all the features contained in this entity

    get_features_names(self)
        Return all the names of the features of this entity

    """

    def __init__(self, attr):
        """
        Parameters
        ----------
        attr:    dict
            Dictionary with the required attributes
        """
        self.name = attr.get('name')
        self.state_dimension = attr.get('state_dimension')
        self.operations, self.features_name = self.compute_hs_operations(attr.get('initial_state'))

    def compute_hs_operations(self, operations):
        """
        Parameters
        ----------
        operations:    dict
            Dictionary specifing an operation to be performed as part of the pipeline to compute the hidden states of a given entity type nodes.
        """

        hs_operations = []
        all_inputs = set()
        all_outputs = set()
        for op in operations:
            all_inputs.update(op.get('input'))  # keep track of all the necessary features

            # keep track of the outputs
            if 'output_name' in op:
                all_outputs.add(op.get('output_name'))

            # create the new operation
            type_update = op.get('type')
            if type_update == 'neural_network':
                hs_operations.append(Feed_forward_operation(op, model_role='hs_creation'))

            elif type_update == 'build_state':
                hs_operations.append(Build_state(op, self.name, self.state_dimension))

        # remove all the references in the inputs that refer to a previous output
        final_inputs = all_inputs.difference(all_outputs)
        return hs_operations, list(final_inputs)

    def get_entity_total_feature_size(self):
        total = 0
        for f in self.features:
            total += f.size

        return total

    def get_features_names(self):
        return [f.name for f in self.features]


class Message_Passing:
    """
    Class that represents the message passing to a single destination entity (with its possible source entities)

    Attributes
    ----------
    destination_entity:     str
        Name of the destination entity
    source_entities:      [array]
        Array of Mp_source_entity object that define a single message passing (between one entity to a destination entity)
    aggregations:     [array]
        Array of aggregation operations that define the aggregation function
    update:     object
        Object with the update model to be used

    Methods:
    --------
    create_update(self,dict)
        Create a model to update the hidden state of the destination entity

    create_aggregations(self, attrs)
        Creates the a set of operations that themselves define the pipeline of operations that constitute the aggregation strategy

    get_instance_info(self)
        Returns the information of this MP
    """

    def __init__(self, m):
        """
        Parameters
        ----------
        m:    dict
            Dictionary with the required attributes
        """

        self.destination_entity = m.get('destination_entity')
        self.source_entities = [Mp_source_entity(s) for s in m.get('source_entities')]

        self.aggregations, self.aggregations_global_type = self.create_aggregations(m.get('aggregation'))
        self.update = self.create_update(m.get('update', {'type': 'direct_assignment'}))

    def create_update(self, u):
        """
        Parameters
        ----------
        u:    dict
            Dictionary with the required attributes for the update
        """

        type_update = u.get('type')
        if type_update == 'direct_assignment':
            return None

        # it is using a neural network
        else:
            first_layer_type = u['architecture'][0]['type_layer']
            if first_layer_type == 'GRU' or first_layer_type == 'LSTM':
                return RNN_operation(u)
            else:
                return Feed_forward_operation(u, model_role='update')

    def create_aggregations(self, attrs):
        """
        Parameters
        ----------
        attrs:    dict
            Dictionary with the required attributes for the aggregation (defining the set of operations)
        """
        aggregations = []
        single_embedding = None
        multiple_embedding = None
        for attr in attrs:
            type = attr.get('type')
            if type == 'interleave':
                aggregations.append(Interleave_aggr(attr))
                multiple_embedding = True
            elif type == 'neural_network':
                aggregations.append(Feed_forward_operation(attr, model_role='aggregation'))
                single_embedding = True
            elif type == 'concat':
                aggregations.append(Concat_aggr(attr))
                multiple_embedding = True
            elif type == 'sum':
                aggregations.append(Sum_aggr(attr))
                single_embedding = True
            elif type == 'mean':
                aggregations.append(Mean_aggr(attr))
                single_embedding = True
            elif type == 'min':
                aggregations.append(Min_aggr(attr))
                single_embedding = True
            elif type == 'max':
                aggregations.append(Max_aggr(attr))
                single_embedding = True
            elif type == 'std':
                aggregations.append(Std_aggr(attr))
                single_embedding = True
            elif type == 'attention':
                aggregations.append(Attention_aggr(attr))
                single_embedding = True
            elif type == 'edge_attention':
                aggregations.append(Edge_attention_aggr(attr))
                single_embedding = True
            elif type == 'convolution':
                aggregations.append(Conv_aggr(attr))
                single_embedding = True
            else:  # this is for the ordered aggregation
                multiple_embedding = True

        if single_embedding and multiple_embedding:
            print_failure(
                "You cannot combine aggregations which return a sequence of tensors, and aggregations that return a single embedding")

        elif single_embedding:
            return aggregations, 0

        else:
            return aggregations, 1

    def get_instance_info(self):
        return [src.get_instance_info(self.destination_entity) for src in self.source_entities]


class Mp_source_entity:
    """
    Class that represents the information of a source entity for the message passing phase

    Attributes
    ----------
    name:   str
        Name of the source entity
    message_formation:      str
        Array of Operation instances

    Methods:
    --------
    create_message_formation(self, operations)
        Creates an array of Operation instances to define the message creation function (pipeline of operations)

    get_instance_info(self, dst_name)
        Returns a string with the information of the source and destination entities of this mp
    """

    def __init__(self, attr):
        """
        Parameters
        ----------
        attr:    dict
            Dictionary with the data defining this mp (extracted from the model description file)
        """

        self.name = attr.get('name')
        self.message_formation = self.create_message_formation(attr.get('message')) if 'message' in attr else [None]

    def create_message_formation(self, operations):
        """
        Parameters
        ----------
        operations:    array
            Array of operation dictionaries
        """

        result = []
        counter = 0
        for op in operations:
            type = op.get('type')
            if type == 'neural_network':
                result.append(Feed_forward_operation(op, model_role='message_creation_' + str(counter)))

            elif type == 'direct_assignment':
                result.append(None)

            elif type == 'product':
                result.append(Product_operation(op))
            counter += 1
        return result

    def get_instance_info(self, dst_name):
        """
        Parameters
        ----------
        dst_name:    dict
            Name of the destination entity
        """

        return self.name + '_to_' + dst_name
