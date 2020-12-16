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
    get_entity_total_feature_size(self)
        Sum of the dimension of all the features contained in this entity

    get_features_names(self)
        Return all the names of the features of this entity

    add_feature(self, f)
        Adds a feature to this entity

    calculate_hs(self, input)
        Returns the corresponding hs of this entity (tensorflow object)

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
        hs_operations = []
        all_inputs = set()
        all_outputs = set()
        for op in operations:
            all_inputs.update(op.get('input'))  #keep track of all the necessary features

            # keep track of the outputs
            if 'output_name' in op:
                all_outputs.add(op.get('output_name'))

            # create the new operation
            type_update = op.get('type')
            if type_update == 'feed_forward':
                hs_operations.append(Feed_forward_operation(op, model_role='hs_creation'))

            elif type_update == 'build_state':
                hs_operations.append(Build_state(op, self.name, self.state_dimension))

        #remove all the references in the inputs that refer to a previous output
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
    type:   str
        Type of message passing (individual or combined)
    destination_entity:     str
        Name of the destination entity
    Mp_source_entity:      str
        Name of the source entity
    adj_list:     str
        Name from the dataset where the adjacency list can be found
    aggregation:     str
        Type of aggregation strategy
    update:     object
        Object with the update model to be used
    update_type:     str
        Indicates if it uses feed-forward or recurrent update
    formation_type:     str
        Indicates if it used feed-forward or recurrent update
    message_formation:      object (optional)
        Feed forward model for the message formation


    Methods:
    --------
    create_update(self,dict)
        Create a model to update the hidden state of the destination entity

    create_aggregation(self, dict)
        Creates the aggreagation instance

    find_type_of_message_creation(self, type)
        Parses the name of the message creation operation

    get_instance_info(self):
        Returns an array with all the relevant information of this message passing

    set_message_formation(self, message_neural_net, number_extra_parameters = 0)
        Sets the message formation strategy by creating the appropriate model to do so.

    add_message_formation_layer(self, **dict):
        Adds a layer to the message formation model

    set_aggregation(self, aggregation)
        Sets the aggregation strategy for the mp

    set_update_strategy(self, update_type, recurrent_type = 'GRU')
        Sets the update strategy for the mp by creating the appropriate model

    add_update_layer(self, **dict)
        Adds a layer to the update model
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
        if type_update == 'feed_forward':
            return Feed_forward_operation(u, model_role='update')

        elif type_update == 'recurrent_neural_network':
            return RNN_operation(u, model_role= 'update')

        elif type_update == 'direct_assignment':
            return None

    def create_aggregations(self, attrs):
        """
        Parameters
        ----------
        dict:    dict
            Dictionary with the required attributes for the aggregation
        """
        aggregations = []
        single_embedding = None
        multiple_embedding = None
        for attr in attrs:
            type = attr.get('type')
            if type == 'interleave':
                aggregations.append(Interleave_aggr(attr))
                multiple_embedding = True
            elif type == 'feed_forward':
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
            else:   # this is for the ordered aggregation
                multiple_embedding = True


        if single_embedding and multiple_embedding:
            print_failure("You cannot combine aggregations which return a sequence of tensors, and aggregations that return a single embedding")

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
    adj_list:     str
        Name of the adj_list
    message_formation:      str
        Array of Operation instances
    extra_parameter:     int
        Size of the extra parameters, if nany


    Methods:
    --------
    create_message_formation(self, operations)
        Creates an array of Operation instances

    get_instance_info(self, dst_name)
        Returns information of the class

    """
    def __init__(self, attr):
        self.name = attr.get('name')
        self.adj_list = attr.get('adj_list')
        self.message_formation = self.create_message_formation(attr.get('message')) if 'message' in attr else [None]
        self.extra_parameters = attr.get('extra_parameters', 0)

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
            if type == 'feed_forward':
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
        return [self.adj_list, self.name, dst_name, str(self.extra_parameters > 0)]

