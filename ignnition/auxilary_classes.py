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


class Feature:
    """
    A class that represents a feature of an entity

    Attributes
    ----------
    name:    str
        Name of the feature
    size:   int
        Dimension of the feature
    normalization:  str
        Type of normalization to be applied to this feature

    """
    def __init__(self, f):
        """
        Parameters
        ----------
        f:    dict
            Dictionary with the required attributes
        """
        self.name = f.get('name')
        self.size = f.get('size',1)
        self.normalization = f.get('normalization','None')

class Entity:
    """
    A class that represents an entity of the MSMP graph

    Attributes
    ----------
    name:    str
        Name of the feature
    hidden_state_dimension:   int
        Dimension of the hiddens state of these entity node's
    features:  array
        Array with all the feature objects that it contains


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
        self.hidden_state_dimension = attr.get('hidden_state_dimension')
        self.features = [Feature(f) for f in attr.get('features', [])]

        # check that the dimension of the hidden-state is big enough
        dim_features = self.get_entity_total_feature_size()
        if dim_features > self.hidden_state_dimension:
            print_failure("The dimension of the hidden_state of entity '" + self.name + "' is insufficient to fit all its features. It should be at least " + str(dim_features) )


    def get_entity_total_feature_size(self):
        total = 0
        for f in self.features:
            total += f.size

        return total

    def get_features_names(self):
        return [f.name for f in self.features]

    def add_feature(self, f):
        """
        Parameters
        ----------
        f:    Feature
            Feature class instance
        """
        self.features.append(f)


    def calculate_hs(self, input):
        """
        Parameters
        ----------
        input:    dict
            Dictionary with the required attributes
        """
        first = True
        total = 0

        # concatenate all the features
        for feature in self.features:
            name_feature = feature.name
            size = feature.size
            total += size
            with tf.name_scope('add_feature_' + str(name_feature)) as _:
                aux = tf.reshape(input.get(name_feature),
                                                 tf.stack([input.get('num_' + str(self.name)), size]))

                if first:
                    state = aux
                    first = False
                else:
                    state = tf.concat([state, aux], axis=1, name="add_" + name_feature)

        shape = tf.stack([input.get('num_' + self.name), self.hidden_state_dimension - total], axis=0)  # shape (2,)

        # add 0s until reaching the given dimension
        with tf.name_scope('add_zeros_to_' + str(self.name)) as _:
            state = tf.concat([state, tf.zeros(shape)], axis=1, name="add_zeros_" + self.name)
            return state


class Aggregation:
    """
    A class that represents an aggregation operation

    Attributes
    ----------
    type:    str
        Type of aggreagation
    """
    def __init__(self, dict):
        self.type = dict.get('type')
        self.output_name = dict.get('output_name', None)


class Sum_aggr(Aggregation):
    """
    A subclass that represents the Sum aggreagtion operation

    Methods:
    ----------
    calculate_input(self, src_input)
        Caclulates the result of applying the sum aggregation
    """

    def __init__(self, dict):
        super(Sum_aggr, self).__init__(dict)

    def calculate_input(self, comb_src_states, comb_dst_idx, num_dst):
        """
        Parameters
        ----------
        src_input:    tensor
           Source entity hs
        """
        src_input = tf.math.unsorted_segment_sum(comb_src_states, comb_dst_idx, num_dst)
        return src_input

class Mean_aggr(Aggregation):
    """
    A subclass that represents the average aggreagtion operation

    Methods:
    ----------
    calculate_input(self, src_input)
        Caclulates the result of applying the sum aggregation
    """

    def __init__(self, dict):
        super(Mean_aggr, self).__init__(dict)

    def calculate_input(self, comb_src_states, comb_dst_idx, num_dst):
        """
        Parameters
        ----------
        src_input:    tensor
           Source entity hs
        """
        neighbours_mean = tf.math.unsorted_segment_mean(comb_src_states, comb_dst_idx, num_dst)
        return neighbours_mean

class Max_aggr(Aggregation):
    """
    A subclass that represents the Sum aggreagtion operation

    Methods:
    ----------
    calculate_input(self, src_input)
        Caclulates the result of applying the sum aggregation
    """

    def __init__(self, dict):
        super(Max_aggr, self).__init__(dict)

    def calculate_input(self, comb_src_states, comb_dst_idx, num_dst):
        """
        Parameters
        ----------
        src_input:    tensor
           Source entity hs
        """
        src_input = tf.math.unsorted_segment_max(comb_src_states, comb_dst_idx, num_dst)
        return src_input

class Min_aggr(Aggregation):
    """
    A subclass that represents the Sum aggreagtion operation

    Methods:
    ----------
    calculate_input(self, src_input)
        Caclulates the result of applying the sum aggregation
    """

    def __init__(self, dict):
        super(Min_aggr, self).__init__(dict)

    def calculate_input(self, comb_src_states, comb_dst_idx, num_dst):
        """
        Parameters
        ----------
        src_input:    tensor
           Source entity hs
        """
        src_input = tf.math.unsorted_segment_min(comb_src_states, comb_dst_idx, num_dst)
        return src_input

# FINISH THIS ONE
class Std_aggr(Aggregation):
    """
    A subclass that represents the Sum aggreagtion operation

    Methods:
    ----------
    calculate_input(self, src_input)
        Caclulates the result of applying the sum aggregation
    """
    def __init__(self, dict):
        super(Std_aggr, self).__init__(dict)

    def calculate_input(self, comb_src_states, comb_dst_idx, num_dst):
        """
        Parameters
        ----------
        src_input:    tensor
           Source entity hs
        """
        src_input = tf.math.unsorted_segment_sum(comb_src_states, comb_dst_idx, num_dst)
        return src_input

class Attention_aggr(Aggregation):
    """
    A subclass that represents the Sum aggreagtion operation

    Methods:
    ----------
    calculate_input(self, comb_src_states, comb_dst_idx, dst_states, comb_seq, num_dst, kernel1, kernel2, attn_kernel)
        Caclulates the result of applying the attention mechanism
    """
    def __init__(self, dict):
        super(Attention_aggr, self).__init__(dict)
        self.weight_initialization = dict.get('weight_initialization', None)

    def calculate_input(self, comb_src_states, comb_dst_idx, dst_states, comb_seq, num_dst, node_kernel, attn_kernel):
        """
        Parameters
        ----------
        comb_src_states:    tensor
            Source hs
        comb_dst_idx:   tensor
            Destination indexes to be combined with (src -> dst)
        dst_states: tensor
            Destination hs
        comb_seq:   tensor

        num_dst:    int
            Number of destination entity nodes
        kernel1:    tf object
            Kernel1 object to transform the source's hs shape
        kernel2:    tf.object
            Kernel2 object to transform the destination's hs shape
        attn_kernel:    tf.object
            Attn_kernel object
        """

        # obtain the source states  (NxF1)
        h_src = tf.identity(comb_src_states)

        # dst_states <- (N x F2)
        #F2 = int(self.dimensions[mp.destination_entity])

        # new number of features (right now set to F1, but could be different)
        #F_ = F1

        # node_kernel = F1 x F1 (we could change the output dimension)
        # transformed_states_sources = NxF1 X F1xF1 = NxF1
        transformed_states_sources = K.dot(h_src, node_kernel) # (W h_i for every source)

        # node_kernel = F2 x F1 (we change the shape of the output hidden state to the same of the source)
        # transformed_states_dest = NxF2 X F2xF1 = NxF1
        dst_states_2 = tf.gather(dst_states, comb_dst_idx)
        transformed_states_dest = K.dot(dst_states_2, node_kernel)  # NxF1   (W h_i for every dst)

        # concat source and dest for each edge
        attention_input = tf.concat([transformed_states_sources, transformed_states_dest], axis=1)  # Nx2F1

        # apply the attention weight vector    (N x 2F1) * (2F1 x 1) = (N x 1)
        # atnn_kernel = 2F1 x 1
        attention_input = K.dot(attention_input, attn_kernel)  # Nx1

        # apply the non linearity
        attention_input = tf.keras.layers.LeakyReLU(alpha=0.2)(attention_input)

        # reshape into a matrix where every row is a destination node and every column is one of its neighbours
        ids = tf.stack([comb_dst_idx, comb_seq], axis=1)
        max_len = tf.reduce_max(comb_seq) + 1
        shape = tf.stack([num_dst, max_len, 1])
        aux = tf.scatter_nd(ids, attention_input, shape)

        # apply softmax to it (by rows)
        coeffients = tf.keras.activations.softmax(aux, axis=0)

        # sum them all together using the coefficients (average)
        final_coeffitients = tf.gather_nd(coeffients, ids)
        weighted_inputs = comb_src_states * final_coeffitients

        src_input = tf.math.unsorted_segment_sum(weighted_inputs, comb_dst_idx,
                                                 num_dst)
        return src_input


class Edge_attention_aggr(Aggregation):
    """
    A subclass that represents the Sum aggreagtion operation

    Methods:
    ----------
    calculate_input(self, comb_src_states, comb_dst_idx, dst_states, comb_seq, num_dst, kernel1, kernel2, attn_kernel)
        Caclulates the result of applying the attention mechanism
    """

    def __init__(self, op):
        super(Edge_attention_aggr, self).__init__(op)
        del op['type']
        self.aggr_model = Feed_forward_operation(op, model_role='edge_attention')


    def get_model(self):
        return self.aggr_model.model

    def calculate_input(self, comb_src_states, comb_dst_idx, num_dst, weights):
        # apply the attention mechanism
        weighted_inputs =  weights * comb_src_states
        # sum by destination nodes
        src_input = tf.math.unsorted_segment_sum(weighted_inputs, comb_dst_idx, int(num_dst))
        return src_input


class Conv_aggr(Aggregation):
    """
    A subclass that represents the Convolution aggreagtion operation

    Methods:
    ----------
    calculate_input(self, comb_src_states, comb_dst_idx, dst_states, num_dst, kernel)
        Caclulates the result of applying the convolution mechanism
    """
    def __init__(self, attr):
        super(Conv_aggr, self).__init__(attr)
        self.activation_function = attr.get('activation_function', 'relu')
        self.weight_initialization = attr.get('weight_initialization', None)

    def calculate_input(self, comb_src_states, comb_dst_idx, dst_states, num_dst, kernel):
        """
        Parameters
        ----------
        comb_src_states:    tensor
            Source hs
        comb_dst_idx:   tensor
            Destination indexes to be combined with (src -> dst)
        dst_states: tensor
            Destination hs

        num_dst:    int
            Number of destination entity nodes
        kernel:    tf object
            Kernel1 object to transform the source's hs shape
        """

        # CONVOLUTION: h_i^t = SIGMA(SUM_N(i) (1 / (sqrt(deg(i)) * sqrt(deg(j))) * w * x_j^(t-1))
        # = h_i^t = SIGMA( 1 / sqrt(deg(i)) * SUM_N(i) (1 / (sqrt(deg(j))) * w * x_j^(t-1))
        # implemented: h_i^t = SIGMA(1 / sqrt(deg(i)) * SUM_N(i) w * x_j^(t-1))

        # comb_src_states = N x F    kernel = F x F
        weighted_input = tf.linalg.matmul(comb_src_states, kernel)

        # normalize each input dividing by sqrt(deg(j)) (only applies if they are from the same entity as the destination node)
        # ??

        # each destination sums all its neighbours
        neighbours_sum = tf.math.unsorted_segment_sum(weighted_input, comb_dst_idx, num_dst)

        # obtain the degrees of each dst_node considering only the entities involved
        dst_deg = tf.math.unsorted_segment_sum(tf.ones_like(comb_dst_idx), comb_dst_idx, num_dst)
        dst_deg = tf.cast(dst_deg, dtype=tf.float32)
        dst_deg = tf.math.sqrt(dst_deg)
        dst_deg = tf.reshape(dst_deg, (-1, 1))

        # normalize the dst_states themselves (divide by their degree)
        dst_states_aux = tf.math.divide_no_nan(dst_states, dst_deg)

        # sum the destination state itself
        total_sum = tf.math.add(neighbours_sum, dst_states_aux)

        #normalize all the values dividing by sqrt(dst_deg)
        normalized_val = tf.math.divide_no_nan(total_sum, dst_deg)

        #normalize by mean and variance  (CHECK) This is the node normalization
        mean = tf.math.reduce_mean(normalized_val)
        var = tf.math.reduce_std(normalized_val)
        normalized_val = (normalized_val - mean) /var

        # apply the non-linearity
        activation_func = getattr(tf.nn, self.activation_function)

        return activation_func(normalized_val)


class Interleave_aggr(Aggregation):
    """
    A subclass that represents the Sum aggreagtion operation

    Methods:
    ----------
    calculate_input(self, src_input, indices)
        Caclulates the result of applying the interleave mechanism
    """

    def __init__(self, dict):
        super(Interleave_aggr, self).__init__(dict)
        self.combination_definition = dict.get('interleave_definition')


    def calculate_input(self, src_input, indices):
        """
        Parameters
        ----------
        src_input:    tensor
            Combined sources hs
        indices:    tensor
            Indices to reorder for the interleave
        """

        # destinations x max_of_sources_to_dest_concat x dim_source ->  (max_of_sources_to_dest_concat x destinations x dim_source)
        src_input = tf.transpose(src_input, perm=[1, 0, 2])
        indices = tf.reshape(indices, [-1, 1])

        src_input = tf.scatter_nd(indices, src_input,
                                  tf.shape(src_input, out_type=tf.int64))

        # (max_of_sources_to_dest_concat x destinations x dim_source) -> destinations x max_of_sources_to_dest_concat x dim_source
        src_input = tf.transpose(src_input, perm=[1, 0, 2])


        # Problem of 0s in between. We need to compress everything leaving the 0s on the right. Then recalculate the real len of each one.
        # Talk to Pere about this. Arnau had the same mistake.


        return src_input


class Concat_aggr(Aggregation):
    """
    A subclass that represents the Concat aggreagtion operation

    Attributes
    ----------
    concat_axis:    str
        Axis to concatenate the input

    """

    def __init__(self, attr):
        super(Concat_aggr, self).__init__(attr)
        self.concat_axis = int(attr.get('concat_axis'))

class Message_Passing:
    """
    Class that represents the message passing to a single destination entity (with its possible source entities)

    Attributes
    ----------
    type:   str
        Type of message passing (individual or combined)
    destination_entity:     str
        Name of the destination entity
    source_entity:      str
        Name of the source entity
    adj_vector:     str
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
        self.source_entities = [Source_Entity(s) for s in m.get('source_entities')]

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

class Source_Entity:
    """
    Class that represents the information of a source entity for the message passing phase

    Attributes
    ----------
    name:   str
        Name of the source entity
    adj_vector:     str
        Name of the adj_vector
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
        self.adj_vector = attr.get('adj_vector')
        self.message_formation = self.create_message_formation(attr.get('message')) if 'message' in attr else [None]
        self.extra_parameters = attr.get('extra_parameters')

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
        return [self.adj_vector, self.name, dst_name, str(self.extra_parameters > 0)]


class Recurrent_Cell:
    """
    Class that represents an RNN model

    Attributes
    ----------
    type:    str
        Type of recurrent model to be used
    params:     dict
        Additional parameters

    Methods:
    --------
    get_tensorflow_object(self, destination_dimension
        Returns a tensorflow object with of this recurrent type with the destination_dimension as the number of units

    perform_unsorted_update(self, model, src_input, old_state)
        Updates the hidden state using the result of applying an input to the model obtained (order doesn't matter)

    perform_sorted_update(self,model, src_input, dst_name, old_state, final_len, num_dst )
        Updates the hidden state using the result of applying an input to the model obtained (order matters)

    """

    def __init__(self, type, parameters):
        """
        Parameters
        ----------
        type:    str
            Type of recurrent model to be used
        parameters:    dict
           Additional parameters of the model
        """
        self.type = type
        self.parameters = parameters
        self.trainable = self.parameters.get('trainable', 'True')
        self.trainable = 'True' == self.trainable

        for k, v in self.parameters.items():
            if v == 'None':
                self.parameters[k] = None

    def get_tensorflow_object(self, destination_dimension):
        """
        Parameters
        ----------
        destination_dimension:    int
            Number of units that the recurrent cell will have
        """
        self.parameters['units'] = destination_dimension
        try:
            c_ = getattr(tf.keras.layers, self.type + 'Cell')
        except:
            print_failure("Error when trying to define a RNN of type '" + self.type + "' since this type does not exist. Check the valid RNN cells that Keras allow to define.")

        try:
            layer = c_(**self.parameters)
            layer.trainable = self.trainable
        except:
            print_failure(
                "Error when creating the RNN of type '" + self.type + "' since invalid parameters were passed. Check the documentation to check which parameters are acceptable or check the spelling of the parameters' names.")

        return layer

    def perform_unsorted_update(self, model, src_input, old_state, dst_dim):
        """
        Parameters
        ----------
        model:    object
            Update model
        src_input:  tensor
            Input for the update operation
        old_state:  tensor
            Old hs of the destination entity
        """
        src_input = tf.ensure_shape(src_input, [None, dst_dim])
        new_state, _ = model(src_input, [old_state])
        return new_state

    def perform_sorted_update(self,model, src_input, dst_name, old_state, final_len):
        """
        Parameters
        ----------
        model:    object
            Update model
        src_input:  tensor
            Input for the update operation
        dst_name:   str
            Destination entity name
        old_state:  tensor
            Old hs of the destination entity
        final_len:  tensor
            Number of source nodes for each destination
        num_dst:    int
            Number of destination nodes
        """
        gru_rnn = tf.keras.layers.RNN(model, name=str(dst_name) + '_update')
        final_len.set_shape([None])
        new_state = gru_rnn(inputs = src_input, initial_state = old_state, mask=tf.sequence_mask(final_len))
        return new_state


class Feed_forward_Layer:
    """
    Class that represents a layer of a feed_forward neural network

    Attributes
    ----------
    type:    str
        Type of recurrent model to be used
    params:     dict
        Additional parameters


    Methods:
    --------
    get_tensorflow_object(self, l_previous)
        Returns a tensorflow object of the containing layer, and sets its previous layer.

    get_tensorflow_object_last(self, l_previous, destination_units)
        Returns a tensorflow object of the last layer of the model, and sets its previous layer and the number of output units for it to have.

    """
    def __init__(self, type, parameters):
        """
        Parameters
        ----------
        type:    str
            ?
        parameters:    dict
            Additional parameters of the model
        """
        self.type = type

        if 'kernel_regularizer' in parameters:
            try:
                parameters['kernel_regularizer'] = tf.keras.regularizers.l2(float(parameters.get('kernel_regularizer')))
            except:
                print_failure("The kernel regularizer parameter '" + str(parameters.get('kernel_regularizer')) + "' in layer of type " + self.type + " is invalid. Please make sure it is a numerical value.")

        if 'activation' in parameters:
            activation = parameters.get('activation')
            if activation == 'None':
                parameters['activation'] = None
            else:
                try:
                    parameters['activation'] = getattr(tf.nn, activation)
                except:
                    print_failure("The activation '" + activation + "' is not a valid function from the tf.nn library. Please check the documentation and the spelling of the function.")


        self.trainable = parameters.get('trainable', 'True')
        parameters['trainable'] = 'True' == self.trainable
        self.parameters = parameters

    def get_tensorflow_object(self, l_previous):
        """
        Parameters
        ----------
        l_previous:    object
            Previous layer of the architecture
        """
        try:
            c_ = getattr(tf.keras.layers, self.type)
        except:
            print_failure("The layer of type '" + self.type + "' is not a valid tf.keras layer. Please check the documentation to write the correct way to define this layer. ")

        try:
            layer = c_(**self.parameters)(l_previous)
        except:
            parameters_string = ''
            for k,v in self.parameters.items():
                parameters_string += k + ': ' + v + '\n'
            print_failure("One of the parameters passed to the layer of type '" + self.type +  "' is incorrect. \n " +
                        "You have defined the following parameters: \n" + parameters_string)

        return layer


    def get_tensorflow_object_last(self, l_previous, destination_units):
        """
        Parameters
        ----------
        l_previous:    object
            Previous layer of the architecture
        destination_dimension:    int
            Number of units that the recurrent cell will have
        """
        try:
            c_ = getattr(tf.keras.layers, self.type)
        except:
            print_failure("The layer of type '" + self.type + "' is not a valid tf.keras layer. Please check the documentation to write the correct way to define this layer. ")

        self.parameters['units'] = destination_units

        try:
            layer = c_(**self.parameters)(l_previous)
        except:
            parameters_string = ''
            for k,v in self.parameters.items():
                parameters_string += k + ': ' + v + '\n'
            print_failure("One of the parameters passed to the layer of type '" + self.type +  "' is incorrect. \n " +
                        "You have defined the following parameters: \n" + parameters_string)

        return layer


class Feed_forward_model:
    """
    Class that represents a feed_forward neural network

    Attributes:
    ----------
    layers:    array
        Layers contained in this feed-forward
    counter:    int
        Counts the current number of layers


    Methods:
    --------
    construct_tf_model(self, var_name, input_dim, dst_dim = None, is_readout = False, dst_name = None)
        Returns the corresponding neural network object

    add_layer(self, **l)
        Add a layer using a dictionary as input

    add_layer_aux(self, l)
        Add a layer

    """

    def __init__(self, model, model_role, n_extra_params = 0):
        """
        Parameters
        ----------
        model:    dict
            Information regarding the architecture of the feed-forward
        """

        self.layers = []
        self.counter = 0
        self.extra_params = n_extra_params

        if 'architecture' in model:
            dict = model['architecture']
            for l in dict:
                type_layer = l['type_layer']  # type of layer
                if 'name' not in l:
                    l['name'] = 'layer_' + str(self.counter) + '_' + type_layer + '_' + str(model_role)
                del l['type_layer']  # leave only the parameters of the layer

                layer = Feed_forward_Layer(type_layer, l)
                self.layers.append(layer)
                self.counter += 1


    def construct_tf_model(self, var_name, input_dim, dst_dim = None, is_readout = False, dst_name = None):
        """
        Parameters
        ----------
        var_name:    str
            Name of the variables
        input_dim:  int
            Dimension of the input of the model
        dst_dim:  int
            Dimension of the destination hs if any
        is_readout: bool
            Is readout?
        dst_name:   str
            Name of the destination entity
        """
        setattr(self, str(var_name) + "_layer_" + str(0),
                tf.keras.Input(shape=(input_dim)))

        layer_counter = 1
        n = len(self.layers)

        for j in range(n):
            l = self.layers[j]
            l_previous = getattr(self, str(var_name) + "_layer_" + str(layer_counter - 1))
            try:
                # if it's the last layer and we haven't defined an output dimension
                if j==(n-1) and dst_dim is not None:
                    layer_model = l.get_tensorflow_object_last(l_previous, dst_dim)
                else:
                    layer_model = l.get_tensorflow_object(l_previous)

                setattr(self, str(var_name) + "_layer_" + str(layer_counter), layer_model)

            except:
                if dst_dim is None: #message_creation
                    if is_readout:
                        print_failure('The layer ' + str(
                                layer_counter) + ' of the readout is not correctly defined. Check keras documentation to make sure all the parameters are correct.')
                    else:
                        print_failure('The layer ' + str(
                            layer_counter) + ' of the message creation neural network in the message passing to ' + str(
                            dst_name) +' is not correctly defined. Check keras documentation to make sure all the parameters are correct.')

                else:
                    print_failure('The layer ' + str(
                            layer_counter) + ' of the update neural network in message passing to ' + str(dst_name) +
                        ' is not correctly defined. Check keras documentation to make sure all the parameters are correct.')

            output_shape = int(layer_model.shape[1])
            layer_counter += 1

        model = tf.keras.Model(inputs=getattr(self, str(var_name) + "_layer_" + str(0)),
                               outputs=getattr(self, str(var_name) + "_layer_" + str(
                                   layer_counter - 1)))
        return [model, output_shape]

class Operation():
    """
    Class that represents an operation to be used in the readout phase

    Attributes:
    ----------
    type:    str
        Type of operation
    input:    array
        Array of the input names
    output_name:    int
        Name to save the output of the operation with

    """

    def __init__(self, op):
        self.type = op.get('type')
        self.output_name = None
        self.output_label = None

        if 'output_label' in op:
            self.output_label = op['output_label']

        if 'output_name' in op:
            self.output_name = op.get('output_name')

        if 'input' in op:
            self.input = op.get('input')


class Product_operation(Operation):
    """
    Subclass of Readout_operation that represents the product operation

    Attributes:
    ----------
    type_product:    str
        Type of product to be used
    output_name:    int
        Name to save the output of the operation with

    """
    def __init__(self, op):
        super(Product_operation, self).__init__(op)
        self.type_product = op.get('type_product')

    def calculate(self, product_input1, product_input2):
        """
        Parameters
        ----------
        product_input1:    tensor
           Input 1
        product_input2:    tensor
           Input 2
        """
        try:
            if self.type_product == 'dot_product':
                result = tf.tensordot(product_input1, product_input2, axes=[[1],[1]])

                # the correct values are in the diagonal (IMPROVE THIS)
                # This does the dot product row by row (so independently for each adjacency)
                result = tf.linalg.tensor_diag_part(result)
                result = tf.expand_dims(result, axis=-1)

            elif self.type_product == 'element_wise':
                result = tf.math.multiply(product_input1, product_input2)

            elif self.type_product == 'mat_mult':
                result = tf.tensordot(product_input1, product_input2, axes=[[2],[1]])
                result = tf.squeeze(result, axis=2)

            return result

        except:
            print_failure('The product operation between ' + product_input1 + ' and ' + product_input2 + ' failed. Check that the dimensions are compatible.')
            sys.exit(1)

class Pooling_operation(Operation):
    """
    Subclass of Readout_operation that represents the product operation

    Attributes:
    ----------
    type_pooling:    str
        Type of pooling to be used

    Methods:
    --------
    calculate(self, input)
        Applies the pooling operation to an input
    """

    def __init__(self, operation):
        """
        Parameters
        ----------
        output:    dict
            Dictionary with the readout_model parameters
        """

        super(Pooling_operation, self).__init__(operation)
        self.type_pooling = operation.get('type_pooling')

    def calculate(self, pooling_input):
        """
        Parameters
        ----------
        pooling_input:    tensor
           Input
        """

        if self.type_pooling == 'sum':
            result = tf.reduce_sum(pooling_input, 0)
            result = tf.reshape(result, [-1] + [result.shape.as_list()[0]])

        elif self.type_pooling == 'mean':
            result = tf.reduce_mean(pooling_input, 0)
            result = tf.reshape(result, [-1] + [result.shape.as_list()[0]])

        elif self.type_pooling == 'max':
            result = tf.reduce_max(pooling_input, 0)
            result = tf.reshape(result, [-1] + [result.shape.as_list()[0]])

        return result


class Feed_forward_operation(Operation):
    """
    Subclass of Readout_operation that represents the readout_nn operation

    Attributes:
    ----------
    input:    array
        Array of input names
    architecture: object
        Neural network object
    output_name:    int
        Name to save the output of the operation with
    """

    def __init__(self, op, model_role):
        super(Feed_forward_operation, self).__init__(op)

        # we need somehow to find the number of extra_parameters beforehand
        self.model = Feed_forward_model({'architecture': op.get('architecture')}, model_role=model_role)


class RNN_operation(Operation):
    def __init__(self, op, model_role):
        super(RNN_operation, self).__init__(op)

        if 'input' in op:
            self.input = op.get('input')

        del op['type']
        self.recurrent_type = op.get('recurrent_type')
        del op['recurrent_type']

        # we need somehow to find the number of extra_parameters beforehand
        self.model = Recurrent_Cell(self.recurrent_type, op)

class Extend_adjacencies(Operation):
    """
    Subclass of Readout_operation that represents the extend_adjacencies operation

    Attributes:
    ----------
    adj_list:    str
        Adjacency list to be used
    output_name:    int
        Name to save the output of the operation with

    Methods:
    --------
    calculate(self, src_states, adj_src, dst_states, adj_dst)
        Applies the extend_adjacency operation to two inputs
    """

    def __init__(self, op):
        super(Extend_adjacencies, self).__init__({'type': op['type'], 'input': op['input']})
        self.adj_list = op['adj_list']
        self.output_name = [op.get('output_name_src'), op.get('output_name_dst')]


    def calculate(self, src_states, adj_src, dst_states, adj_dst):
        """
        Parameters
        ----------
        src_states:    tensor
           Input 1
        adj_src:    tensor
            Adj src -> dest
        dst_states:     tensor
            Input 2
        adj_dst:    tensor
            Adj dst -> src
        """

        # obtain the extended input (by extending it to the number of adjacencies between them)
        try:
            extended_src = tf.gather(src_states, adj_src)
        except:
            print_failure('Extending the adjacency list ' + str(
                self.adj_list) + ' was not possible. Check that the indexes of the source of the adjacency list match the input given.')

        try:
            extended_dst = tf.gather(dst_states, adj_dst)
        except:
            print_failure('Extending the adjacency list ' + str(
                self.adj_list) + ' was not possible. Check that the indexes of the destination of the adjacency list match the input given.')

        return extended_src, extended_dst


class Weight_matrix:
    def __init__(self, nn):
        self.nn_name = nn.get('nn_name')
        self.weight_dimensions = nn.get('weight_dimensions')
        self.trainable = nn.get('trainable', 'True')
        self.trainable = self.trainable == 'True'