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
from keras import backend as K
from ignnition.auxilary_classes import *
from functools import reduce

import numpy as np


class Gnn_model(tf.keras.Model):
    """
    Class that represents the final GNN
    Methods
    ----------
    call(self, input, training=False)
        Performs the GNN's action
    get_global_var_or_input(self, var_name, input)
        Obtains the global variable with var_name if exists, or the corresponding input
    save_global_variable(self, var_name, var_value)
        Save the global variable with var_name and with the corresponding value
    get_global_variable(self, var_name)
        Obtains the global variable with the corresponding var_name
    """

    def __init__(self, model_info):
        super(Gnn_model, self).__init__()
        self.model_info = model_info
        self.dimensions = self.model_info.get_input_dimensions()
        self.instances_per_stage = self.model_info.get_mp_instances()
        with tf.name_scope('model_initializations') as _:
            for stage in self.instances_per_stage:
                # here we save the input for each of the updates that will be done at the end of the stage
                for message in stage[1]:
                    dst_name = message.destination_entity
                    with tf.name_scope('message_creation_models') as _:
                        # acces each source entity of this destination
                        for src in message.source_entities:
                            operations = src.message_formation
                            src_name = src.name
                            counter = 0
                            output_shape = int(
                                self.dimensions.get(src_name))  # default if operation is direct_assignation
                            for operation in operations:
                                if operation is not None:
                                    if operation.type == 'feed_forward':
                                        var_name = src_name + "_to_" + dst_name + '_message_creation_' + str(counter)

                                        # Find out the dimension of the model
                                        input_nn = operation.input
                                        input_dim = 0
                                        for i in input_nn:
                                            if i == 'hs_source':
                                                input_dim += int(self.dimensions.get(src_name))
                                            elif i == 'hs_dst':
                                                input_dim += int(self.dimensions.get(dst_name))
                                            elif i == 'edge_params':
                                                input_dim += int(src.extra_parameters)  # size of the extra parameter
                                            else:
                                                dimension = self.get_global_variable(i + '_dim')
                                                input_dim += dimension

                                        model, output_shape = operation.model.construct_tf_model(var_name, input_dim)

                                        self.save_global_variable(var_name, model)

                                        # Need to keep track of the output dimension of this one, in case we need it for a new model
                                        if operation.output_name is not None:
                                            self.save_global_variable(operation.output_name + '_dim', output_shape)

                                    elif operation.type == 'product':
                                        if operation.type_product == 'dot_product':
                                            output_shape = 1


                            self.save_global_variable("final_message_dim_" + src.adj_vector, output_shape)

                            counter += 1

                    with tf.name_scope('aggregation_models') as _:
                        aggregation = message.aggregation
                        F_dst = int(self.dimensions.get(dst_name))
                        F_src = int(output_shape)

                        if aggregation.type == 'attention':
                            self.kernel1 = self.add_weight(shape=(F_src, F_src),
                                                           initializer=aggregation.weight_initialization)
                            self.kernel2 = self.add_weight(shape=(F_dst, F_src),
                                                           initializer=aggregation.weight_initialization)
                            self.attn_kernel = self.add_weight(shape=(2 * F_dst, 1),
                                                               initializer=aggregation.weight_initialization)

                        elif aggregation.type == 'edge_attention':
                            # create a neural network that takes the concatenation of the source and dst message and results in the weight
                            message_dimensionality = F_src + F_dst
                            var_name = 'edge_attention_' + src_name + '_to_' + dst_name  # choose the src_name of the last?
                            model, _ = aggregation.get_model().construct_tf_model(var_name=var_name,
                                                                                  input_dim=message_dimensionality)
                            self.save_global_variable(var_name, model)

                        elif aggregation.type == 'convolution':
                            if F_dst != F_src:
                                tf.compat.v1.logging.error(
                                    'IGNNITION: When doing the a convolution, both the dimension of the messages sent and the destination hidden states should match. '
                                    'In this case, however,the dimensions are ' + F_src + ' and ' + F_dst + ' of the source and destination respectively.')
                                sys.exit(1)

                            self.conv_kernel = self.add_weight(shape=(F_dst, F_dst),
                                                               initializer=aggregation.weight_initialization)


                    # -----------------------------
                    # Creation of the update models
                    with tf.name_scope('update_models') as _:
                        update_model = message.update

                        # ------------------------------
                        # create the recurrent update models
                        if update_model.type == "recurrent_neural_network":
                            recurrent_cell = update_model.model
                            try:
                                recurrent_instance = recurrent_cell.get_tensorflow_object(self.dimensions.get(dst_name))
                                self.save_global_variable(dst_name + '_update', recurrent_instance)
                            except:
                                tf.compat.v1.logging.error(
                                    'IGNNITION: The definition of the recurrent cell in message passsing to ' + message.destination_entity +
                                    ' is not correctly defined. Check keras documentation to make sure all the parameters are correct.')
                                sys.exit(1)


                        # ----------------------------------
                        # create the feed-forward upddate models
                        # This only makes sense with aggregation functions that preserve one single input (not sequence)
                        else:
                            model = update_model.model
                            source_entities = message.source_entities
                            var_name = dst_name + "_ff_update"

                            with tf.name_scope(dst_name + '_ff_update') as _:
                                dst_dim = int(self.dimensions.get(dst_name))

                                # calculate the message dimensionality (considering that they all have the same dim)
                                # o/w, they are not combinable
                                message_dimensionality = self.get_global_variable(
                                    "final_message_dim_" + source_entities[0].adj_vector)

                                # if we are concatenating by message
                                aggr = message.aggregation
                                if aggr.type == 'concat' and aggr.concat_axis == 2:
                                    message_dimensionality = reduce(lambda accum, s: accum + int(
                                        getattr(self, "final_message_dim_" + s.adj_vector)),
                                                                    source_entities, 0)

                                input_dim = message_dimensionality + dst_dim  # we will concatenate the sources and destinations

                                model, _ = model.construct_tf_model(var_name, input_dim, dst_dim, dst_name=dst_name)
                                self.save_global_variable(var_name, model)

            # --------------------------------
            # Create the readout model
            readout_operations = self.model_info.get_readout_operations()
            counter = 0
            for operation in readout_operations:
                if operation.type == 'feed_forward':
                    with tf.name_scope("readout_architecture"):
                        input_dim = reduce(lambda accum, i: accum + int(self.dimensions.get(i)), operation.input, 0)
                        model, _ = operation.model.construct_tf_model('readout_model' + str(counter), input_dim,
                                                                      is_readout=True)
                        self.save_global_variable('readout_model_' + str(counter), model)

                    # save the dimensions of the output
                    dimensionality = model.layers[-1].output.shape[1]

                elif operation.type == 'pooling':
                    dimensionality = self.dimensions.get(operation.input[0])

                    # add the new dimensionality to the input_dimensions tensor
                    if operation.output_name is not None:
                        dimensionality = dimensionality

                elif operation.type == 'product':
                    if operation.type_product == 'element_wise':
                        dimensionality = self.dimensions.get(operation.input[0])

                    elif operation.type_product == 'dot_product':
                        dimensionality = 1

                elif operation.type == 'extend_adjacencies':
                    self.dimensions[operation.output_name[0]] = self.dimensions.get(operation.input[0])
                    self.dimensions[operation.output_name[1]] = self.dimensions.get(operation.input[1])

                if operation.type!= 'extend_adjacencies' and operation.output_name is not None:
                    self.dimensions[operation.output_name] = dimensionality

                counter += 1

            # ---------------------------------------------------------------------------
            # CREATE ANY OTHER WEIGHT MATRIX IF ANY
            weight_matrices = self.model_info.get_weight_matrices()

            for w in weight_matrices:
                with tf.name_scope("weight_matrix"):
                    dimensions = w.weight_dimensions

                    # create the trainable mask
                    mask = tf.compat.v1.get_variable(
                        w.nn_name,
                        shape=[dimensions[0], dimensions[1]],
                        trainable=w.trainable,
                        initializer=tf.keras.initializers.Zeros())

                    self.save_global_variable(w.nn_name, mask)

    @tf.function
    def call(self, input, training):
        """
        Parameters
        ----------
        input:    dict
            Dictionary with all the tensors with the input information of the model
        """
        with tf.name_scope('ignnition_model') as _:
            f_ = input.copy()
            for k, v in f_.items():
                if len(tf.shape(v)) == 2 and tf.shape(v)[1] == 1:
                    f_[k] = tf.squeeze(v, axis=-1)
            # -----------------------------------------------------------------------------------
            # HIDDEN STATE CREATION
            entities = self.model_info.entities

            # Initialize all the hidden states for all the nodes.
            with tf.name_scope('hidden_states') as _:
                for entity in entities:
                    with tf.name_scope('hidden_state_' + str(entity.name)) as _:
                        state = entity.calculate_hs(f_)
                        self.save_global_variable(entity.name + '_state', state)
                        self.save_global_variable(entity.name + '_initial_state', state)

            # -----------------------------------------------------------------------------------
            # MESSAGE PASSING PHASE
            with tf.name_scope('message_passing') as _:
                for j in range(self.model_info.get_mp_iterations()):

                    with tf.name_scope('iteration_' + str(j)) as _:

                        for stage in self.instances_per_stage:
                            step_name = stage[0]

                            with tf.name_scope(step_name) as _:
                                # given one message from a given step
                                for mp in stage[1]:
                                    dst_name = mp.destination_entity
                                    dst_states = self.get_global_variable(str(dst_name) + '_state')
                                    num_dst = f_['num_' + dst_name]

                                    # with tf.name_scope('mp_to_' + dst_name + 's') as _:
                                    with tf.name_scope(mp.source_entities[0].name + 's_to_' + dst_name + 's') as _:
                                        first_src = True
                                        with tf.name_scope('message') as _:
                                            for src in mp.source_entities:
                                                src_name = src.name

                                                # prepare the information
                                                src_idx, dst_idx, seq = f_.get('src_' + src.adj_vector), f_.get(
                                                    'dst_' + src.adj_vector), f_.get('seq_' + src_name + '_' + dst_name)
                                                src_states = self.get_global_variable(str(src_name) + '_state')

                                                with tf.name_scope(
                                                        'create_message_' + src_name + '_to_' + dst_name) as _:
                                                    self.src_messages = tf.gather(src_states, src_idx)
                                                    self.dst_messages = tf.gather(dst_states, dst_idx)
                                                    message_creation_models = src.message_formation

                                                    # by default, the source hs are the messages
                                                    result = self.src_messages
                                                    counter = 0

                                                    for operation in message_creation_models:
                                                        if operation is not None:  # if it is not direct_assignation
                                                            type_operation = operation.type

                                                            if type_operation == 'feed_forward':
                                                                with tf.name_scope('apply_nn_' + str(counter)) as _:
                                                                    # careful. This name could overlap with another model
                                                                    var_name = src_name + "_to_" + dst_name + '_message_creation_' + str(counter)
                                                                    message_creator = self.get_global_variable(var_name)
                                                                    first = True
                                                                    with tf.name_scope('obtain_input') as _:
                                                                        for i in operation.input:
                                                                            new_input = self.treat_message_function_input(i, src.adj_vector, f_ )
                                                                            # accumulate the results
                                                                            if first:
                                                                                first = False
                                                                                input_nn = new_input
                                                                            else:
                                                                                input_nn = tf.concat(
                                                                                    [input_nn, new_input], axis=1)

                                                                result = message_creator(input_nn)

                                                            elif type_operation == 'product':
                                                                with tf.name_scope(
                                                                        'apply_product_' + str(counter)) as _:
                                                                    with tf.name_scope('obtain_input') as _:

                                                                        product_input1 = self.treat_message_function_input(
                                                                            operation.input[0], src.adj_vector, f_)

                                                                        product_input2 = self.treat_message_function_input(
                                                                            operation.input[1], src.adj_vector, f_)
                                                                    result = operation.calculate(product_input1, product_input2)

                                                            if operation.output_name is not None:
                                                                self.save_global_variable(
                                                                    operation.output_name + '_var', result)

                                                        final_messages = result

                                                        counter += 1

                                                    with tf.name_scope(
                                                            'combine_messages_' + src_name + '_to_' + dst_name) as _:
                                                        ids = tf.stack([dst_idx, seq], axis=1)

                                                        lens = tf.math.unsorted_segment_sum(tf.ones_like(dst_idx),
                                                                                            dst_idx, num_dst)

                                                        # only a few aggregations actually needed to keep the order
                                                        max_len = tf.reduce_max(seq) + 1

                                                        message_dim = int(self.get_global_variable(
                                                            "final_message_dim_" + src.adj_vector))

                                                        shape = tf.stack([num_dst, max_len, message_dim])
                                                        s = tf.scatter_nd(ids, final_messages,
                                                                          shape)  # find the input ordering it by sequence

                                                        aggr = mp.aggregation
                                                        if isinstance(aggr, Concat_aggr):
                                                            with tf.name_scope("concat_" + src_name) as _:
                                                                if first_src:
                                                                    src_input = s
                                                                    final_len = lens
                                                                    first_src = False
                                                                else:
                                                                    src_input = tf.concat([src_input, s],
                                                                                          axis=aggr.concat_axis)
                                                                    if aggr.concat_axis == 1:  # if axis=2, then the number of messages received is the same. Simply create bigger messages
                                                                        final_len += lens

                                                        elif isinstance(aggr, Interleave_aggr):
                                                            with tf.name_scope('add_' + src_name) as _:
                                                                indices_source = f_.get(
                                                                    "indices_" + src_name + '_to_' + dst_name)
                                                                if first_src:
                                                                    first_src = False
                                                                    src_input = s  # destinations x max_of_sources_to_dest x dim_source
                                                                    indices = indices_source
                                                                    final_len = lens
                                                                else:
                                                                    # destinations x max_of_sources_to_dest_concat x dim_source
                                                                    src_input = tf.concat([src_input, s], axis=1)
                                                                    indices = tf.stack([indices, indices_source],
                                                                                       axis=0)
                                                                    final_len = tf.math.add(final_len, lens)


                                                        # if we must aggregate them together into a single embedding (sum, attention, edge_attention, ordered)
                                                        else:
                                                            # obtain the overall input of each of the destinations
                                                            if first_src:
                                                                first_src = False
                                                                src_input = s  # destinations x sources_to_dest x dim_source
                                                                comb_src_states, comb_dst_idx, comb_seq = final_messages, dst_idx, seq  # we need this for the attention and convolutional mechanism
                                                                final_len = lens

                                                            else:
                                                                # destinations x max_of_sources_to_dest_concat x dim_source
                                                                src_input = tf.concat([src_input, s], axis=1)
                                                                comb_src_states = tf.concat(
                                                                    [comb_src_states, final_messages],
                                                                    axis=0)
                                                                comb_dst_idx = tf.concat([comb_dst_idx, dst_idx],
                                                                                         axis=0)

                                                                aux_lens = tf.gather(final_len,
                                                                                     dst_idx)  # lens of each src-dst value
                                                                aux_seq = seq + aux_lens  # sum to the sequences the current length for each dest
                                                                comb_seq = tf.concat([comb_seq, aux_seq], axis=0)

                                                                final_len = tf.math.add(final_len, lens)

                                        # --------------
                                        # perform the actual aggregation
                                        aggr = mp.aggregation

                                        # if ordered, we dont need to do anything. Already in the right shape

                                        with tf.name_scope('aggregation') as _:
                                            if aggr.type == 'sum':
                                                src_input = aggr.calculate_input(comb_src_states, comb_dst_idx, num_dst)

                                            elif aggr.type == 'attention':
                                                src_input = aggr.calculate_input(comb_src_states, comb_dst_idx,
                                                                                 dst_states,
                                                                                 comb_seq, num_dst, self.kernel1,
                                                                                 self.kernel2,
                                                                                 self.attn_kernel)

                                            elif aggr.type == 'edge_attention':
                                                var_name = 'edge_attention_' + src_name + '_to_' + dst_name
                                                edge_att_model = self.get_global_variable(var_name)
                                                comb_dst_states = tf.gather(dst_states,
                                                                            comb_dst_idx)  # the destination state of each adjacency
                                                model_input = tf.concat([comb_src_states, comb_dst_states], axis=1)
                                                weights = edge_att_model(model_input)
                                                src_input = aggr.calculate_input(comb_src_states, comb_dst_idx, num_dst,
                                                                                 weights)

                                            # convolutional aggregation (the messages sent by the destination must have the same shape as the destinations)
                                            elif aggr.type == 'convolution':
                                                src_input = aggr.calculate_input(comb_src_states, comb_dst_idx,
                                                                                 dst_states,
                                                                                 num_dst, self.conv_kernel)

                                            elif aggr.type == 'interleave':
                                                src_input = aggr.calculate_input(src_input, indices)

                                            # save the src_input used for the update
                                            self.save_global_variable('update_lens_' + dst_name, final_len)
                                            self.save_global_variable('update_input_' + dst_name, src_input)

                                # ---------------------------------------
                                # updates
                                with tf.name_scope('updates') as _:
                                    for mp in stage[1]:
                                        aggr = mp.aggregation
                                        dst_name = mp.destination_entity
                                        with tf.name_scope('update_' + dst_name + 's') as _:
                                            update_model = mp.update
                                            src_input = self.get_global_variable('update_input_' + dst_name)
                                            old_state = self.get_global_variable(dst_name + '_state')

                                            # recurrent update
                                            if update_model.type == "recurrent_neural_network":
                                                model = self.get_global_variable(dst_name + '_update')

                                                if aggr.type == 'sum' or aggr.type == 'attention' or aggr.type == 'convolution' or aggr.type == 'edge_attention':
                                                    dst_dim = int(self.dimensions[
                                                                      dst_name])  # should this be the source dimensions??? CHECK
                                                    new_state = update_model.model.perform_unsorted_update(model,
                                                                                                           src_input,
                                                                                                           old_state,
                                                                                                           dst_dim)

                                                # if the aggregation was ordered or concat
                                                else:
                                                    final_len = self.get_global_variable('update_lens_' + dst_name)
                                                    new_state = update_model.model.perform_sorted_update(model,
                                                                                                         src_input,
                                                                                                         dst_name,
                                                                                                         old_state,
                                                                                                         final_len)

                                            # feed-forward update:
                                            # restriction: It can only be used if the aggreagation was not ordered.
                                            elif update_model.type == 'feed_forward':
                                                var_name = dst_name + "_ff_update"
                                                update = self.get_global_variable(var_name)

                                                # now we need to obtain for each adjacency the concatenation of the source and the destination
                                                update_input = tf.concat([src_input, old_state], axis=1)
                                                new_state = update(update_input)

                                            # update the old state
                                            self.save_global_variable(dst_name + '_state', new_state)

            # -----------------------------------------------------------------------------------
            # READOUT PHASE
            with tf.name_scope('readout_predictions') as _:
                readout_operations = self.model_info.get_readout_operations()
                counter = 0
                n = len(readout_operations)
                for j in range(n):
                    operation = readout_operations[j]
                    with tf.name_scope(operation.type) as _:
                        if operation.type == 'feed_forward':
                            first = True
                            for i in operation.input:
                                new_input = self.get_global_var_or_input(i, f_)
                                if first:
                                    nn_input = new_input
                                    first = False
                                else:
                                    nn_input = tf.concat([nn_input, new_input], axis=1)
                            readout_nn = self.get_global_variable('readout_model_' + str(counter))
                            # this is necessary to force batch_size = 1

                            if len(tf.shape(nn_input)) == 1:
                                nn_input = tf.expand_dims(nn_input, axis=0)
                            result = readout_nn(nn_input, training=True)

                        elif operation.type == "pooling":
                            # obtain the input of the pooling operation
                            first = True
                            for input_name in operation.input:
                                aux = self.get_global_var_or_input(input_name, f_)
                                if first:
                                    pooling_input = aux
                                    first = False
                                else:
                                    pooling_input = tf.concat([pooling_input, aux], axis=0)

                            result = operation.calculate(pooling_input)

                        elif operation.type == 'product':
                            product_input1 = self.get_global_var_or_input(operation.input[0], f_)
                            product_input2 = self.get_global_var_or_input(operation.input[1], f_)
                            result = operation.calculate(product_input1, product_input2)

                        # extends the two inputs following the adjacency list that connects them both. CHECK!!
                        elif operation.type == 'extend_adjacencies':
                            adj_src = f_.get('src_' + operation.adj_list)
                            adj_dst = f_.get('dst_' + operation.adj_list)

                            src_states = self.get_global_var_or_input(operation.input[0], f_)
                            dst_states = self.get_global_var_or_input(operation.input[1], f_)

                            extended_src, extended_dst = operation.calculate(src_states, adj_src, dst_states, adj_dst)
                            self.save_global_variable(operation.output_name[0] + '_state', extended_src)
                            self.save_global_variable(operation.output_name[1] + '_state', extended_dst)

                        if operation.type != 'extend_adjacencies':
                            if j == n-1:
                                return result
                            else:
                                self.save_global_variable(operation.output_name + '_state', result)


                    counter += 1

    def get_global_var_or_input(self, var_name, f_):
        """
        Parameters
        ----------
        var_name:    str
            All the features to be used as input
        input:    dict
            Input tensors
        """

        try:
            return self.get_global_variable(var_name + '_state')
        except:
            return f_[var_name]

    # creates a new global variable
    def save_global_variable(self, var_name, var_value):
        """
        Parameters
        ----------
        var_name:    String
            Name of the global variable to save
        var_value:    tensor
            Tensor value of the new global variable
        """
        setattr(self, var_name, var_value)

    def get_global_variable(self, var_name):
        """
        Parameters
        ----------
        var_name:    String
            Name of the global variable to save
        """
        return getattr(self, var_name)

    def treat_message_function_input(self, var_name, adj_vector_name, f_):
        if var_name == 'hs_source':
            new_input = self.src_messages
        elif var_name == 'hs_dst':
            new_input = self.dst_messages
        elif var_name == 'edge_params':
            new_input = tf.cast(
                f_['params_' + adj_vector_name],
                tf.float32)
        else:
            new_input = self.get_global_variable(
                var_name + '_var')

        return new_input