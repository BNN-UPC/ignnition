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

from functools import reduce

import tensorflow as tf

from ignnition.operation_classes import RNNOperation
from ignnition.aggregation_classes import ConcatAggr, InterleaveAggr
from ignnition.utils import save_global_variable, print_failure, get_global_variable, get_global_var_or_input


class GnnModel(tf.keras.Model):
    """
    Class that represents the final GNN
    Methods
    ----------
    call(self, input, training=False)
        Performs the GNN's action
    save_global_variable(self, var_name, var_value)
        Save the global variable with var_name and with the corresponding value
    get_global_variable(self, var_name)
        Obtains the global variable with the corresponding var_name
    """

    def __init__(self, model_info):
        super(GnnModel, self).__init__()
        self.model_info = model_info
        self.dimensions = self.model_info.get_input_dimensions()
        self.instances_per_stage = self.model_info.get_mp_instances()
        self.calculations = {}
        with tf.name_scope('model_initializations') as _:
            entities = model_info.entities
            for entity in entities:
                operations = entity.operations
                counter = 0
                for op in operations:
                    if op.type == 'neural_network':
                        var_name = entity.name + "_hs_creation_" + str(counter)
                        input_dim = op.find_total_input_dim(self.dimensions, self.calculations)

                        model, output_shape = op.model.construct_tf_model(input_dim=input_dim)
                        save_global_variable(self.calculations, var_name, model)

                        # Need to keep track of the output dimension of this one, in case we need it for a new model
                        if op.output_name is not None:
                            save_global_variable(self.calculations, op.output_name + '_dim', output_shape)
                    counter += 1

            stages = self.instances_per_stage
            n_stages = len(stages)
            for idx_stage in range(n_stages):
                msgs_stage = stages[idx_stage][1]
                n_messages = len(msgs_stage)
                # here we save the input for each of the updates that will be done at the end of the stage
                for idx_msg in range(n_messages):
                    message = msgs_stage[idx_msg]
                    dst_name = message.destination_entity
                    with tf.name_scope('message_creation_models') as _:
                        # access each source entity of this destination
                        for src in message.source_entities:
                            operations = src.message_formation
                            src_name = src.name
                            counter = 0
                            output_shape = int(
                                self.dimensions.get(src_name))  # default if operation is direct_assignation
                            for operation in operations:
                                if operation is not None:
                                    if operation.type == 'neural_network':
                                        var_name = src_name + "_to_" + dst_name + '_message_creation_' + str(counter)
                                        input_dim = operation.obtain_total_input_dim_message(self.dimensions,
                                                                                             self.calculations,
                                                                                             dst_name, src)
                                        model, output_shape = operation.model.construct_tf_model(input_dim=input_dim)

                                        save_global_variable(self.calculations, var_name, model)

                                        # Need to keep track of the output dimension of this one,
                                        # in case we need it for a new model
                                        if operation.output_name is not None:
                                            save_global_variable(self.calculations, operation.output_name + '_dim',
                                                                 output_shape)

                                    elif operation.type == 'product':
                                        if operation.type_product == 'dot_product':
                                            output_shape = 1

                            save_global_variable(self.calculations,
                                                 "final_message_dim_" + str(idx_stage) + '_' + str(idx_msg),
                                                 output_shape)

                            counter += 1

                    with tf.name_scope('aggregation_models') as _:
                        aggregations = message.aggregations
                        F_dst = int(self.dimensions.get(dst_name))
                        F_src = int(output_shape)

                        for aggregation in aggregations:
                            # what is the shape if we are using ordered / interleave??
                            output_shape = F_dst  # by default we don't change the shape of the final output

                            if aggregation.type == 'attention':
                                self.node_kernel = self.add_weight(shape=(F_src, F_src),
                                                                   initializer=aggregation.weight_initialization,
                                                                   name='attention_node_kernel')
                                self.attn_kernel = self.add_weight(shape=(2 * F_dst, 1),
                                                                   initializer=aggregation.weight_initialization,
                                                                   name='attention_attn_kernel')

                            elif aggregation.type == 'edge_attention':
                                # create a neural network that takes the concatenation of the source and dst message
                                # and results in the weight
                                message_dimensionality = F_src + F_dst

                                var_name = 'edge_attention_' + src_name + '_to_' + dst_name
                                model, _ = aggregation.get_model().construct_tf_model(input_dim=message_dimensionality)
                                save_global_variable(self.calculations, var_name, model)

                            elif aggregation.type == 'convolution':
                                if F_dst != F_src:
                                    print_failure('When doing the a convolution, both the dimension of the messages '
                                                  'sent and the destination hidden states should match. '
                                                  'In this case, however,the dimensions are ' + F_src + ' and '
                                                  + F_dst + ' of the source and destination respectively.')

                                self.conv_kernel = self.add_weight(shape=(F_dst, F_dst),
                                                                   initializer=aggregation.weight_initialization,
                                                                   name='conv_kernel')

                            elif aggregation.type == 'neural_network':
                                var_name = 'aggr_nn'
                                input_dim = aggregation.find_total_input_dim(self.dimensions, self.calculations)

                                model, output_shape = aggregation.model.construct_tf_model(input_dim=input_dim)

                                save_global_variable(self.calculations, var_name, model)

                            # Need to keep track of the output dimension of this one, in case we need it for a new model
                            if aggregation.output_name is not None:
                                save_global_variable(self.calculations, aggregation.output_name + '_dim', output_shape)

                        save_global_variable(self.calculations,
                                             "final_message_dim_" + str(idx_stage) + '_' + str(idx_msg), output_shape)

                    # -----------------------------
                    # Creation of the update models
                    with tf.name_scope('update_models') as _:
                        update_model = message.update

                        # ------------------------------
                        # create the recurrent update models
                        if update_model is not None and isinstance(update_model, RNNOperation):
                            recurrent_cell = update_model.model
                            try:
                                recurrent_instance = recurrent_cell.get_tensorflow_object(self.dimensions.get(dst_name))
                                save_global_variable(self.calculations, dst_name + '_update', recurrent_instance)
                            except Exception:
                                print_failure('The definition of the recurrent cell in message passing to '
                                              + message.destination_entity + ' is not correctly defined. Check keras '
                                              'documentation to make sure all the parameters are correct.')

                        # ----------------------------------
                        # create the feed-forward upddate models
                        # This only makes sense with aggregation functions that preserve one single input (not sequence)
                        elif update_model is not None:
                            model = update_model.model
                            source_entities = message.source_entities
                            var_name = dst_name + "_ff_update"

                            with tf.name_scope(dst_name + '_ff_update') as _:
                                dst_dim = int(self.dimensions.get(dst_name))

                                # calculate the message dimensionality (considering that they all have the same dim)
                                # o/w, they are not combinable
                                message_dimensionality = get_global_variable(self.calculations,
                                                                             "final_message_dim_" + str(
                                                                                 idx_stage) + '_' + str(idx_msg))

                                # if we are concatenating by message (CHECK!!)
                                if aggregation.type == 'concat' and aggregation.concat_axis == 2:
                                    message_dimensionality = reduce(lambda accum, s: accum + int(
                                        get_global_variable(self.calculations,
                                                            "final_message_dim_" + str(idx_stage) + '_' + str(
                                                                idx_msg))),
                                                                    source_entities, 0)

                                input_dim = message_dimensionality + dst_dim

                                model, _ = model.construct_tf_model(input_dim=input_dim, dst_dim=dst_dim,
                                                                    dst_name=dst_name)
                                save_global_variable(self.calculations, var_name, model)

            # --------------------------------
            # Create the readout model
            readout_operations = self.model_info.get_readout_operations()
            # print(readout_operations)
            counter = 0
            for operation in readout_operations:
                if operation.type == 'neural_network':
                    with tf.name_scope("readout_architecture"):
                        input_dim = operation.find_total_input_dim(self.dimensions, self.calculations)
                        model, _ = operation.model.construct_tf_model(input_dim=input_dim,
                                                                      is_readout=True)
                        save_global_variable(self.calculations, 'readout_model_' + str(counter), model)

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

                elif operation.type == 'concat':
                    dimensionality = 0
                    for inp in operation.input:
                        dimensionality += self.dimensions.get(inp)

                if operation.type != 'extend_adjacencies' and operation.output_name is not None:
                    self.dimensions[operation.output_name] = dimensionality

                counter += 1

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

            # -----------------------------------------------------------------------------------
            # HIDDEN STATE CREATION
            entities = self.model_info.entities

            # Initialize all the hidden states for all the nodes.
            with tf.name_scope('states_creation') as _:
                for entity in entities:
                    with tf.name_scope(str(entity.name)) as _:
                        counter = 0
                        operations = entity.operations
                        for op in operations:
                            if op.type == 'neural_network':
                                with tf.name_scope('apply_nn_' + str(counter)) as _:
                                    # careful. This name could overlap with another model
                                    var_name = entity.name + "_hs_creation_" + str(counter)
                                    hs_creator = get_global_variable(self.calculations, var_name)
                                    output = op.apply_nn(hs_creator, self.calculations, f_)

                                    save_global_variable(self.calculations, op.output_name, output)

                            elif op.type == 'build_state':
                                with tf.name_scope('build_state' + str(counter)) as _:
                                    state = op.calculate_hs(self.calculations, f_)
                                    save_global_variable(self.calculations, entity.name, state)
                                    save_global_variable(self.calculations, entity.name + '_initial_state', state)
                            counter += 1

            # -----------------------------------------------------------------------------------
            # MESSAGE PASSING PHASE
            with tf.name_scope('message_passing') as _:
                for j in range(self.model_info.get_mp_iterations()):

                    with tf.name_scope('iteration_' + str(j)) as _:
                        num_instances_per_stage = len(self.instances_per_stage)
                        for idx_stage in range(num_instances_per_stage):
                            stage = self.instances_per_stage[idx_stage]
                            step_name = stage[0]

                            with tf.name_scope(step_name) as _:
                                # given one message from a given step
                                msgs_stage = stage[1]
                                num_msgs_stage = len(msgs_stage)

                                for idx_msg in range(num_msgs_stage):
                                    mp = msgs_stage[idx_msg]
                                    dst_name = mp.destination_entity
                                    dst_states = get_global_variable(self.calculations, dst_name)
                                    num_dst = f_['num_' + dst_name]

                                    # with tf.name_scope('mp_to_' + dst_name + 's') as _:
                                    with tf.name_scope('MP_to_' + dst_name) as _:

                                        with tf.name_scope('message_phase') as _:
                                            # this is useful to check if there was any message passing
                                            # to do (o/w ignore this MP)
                                            self.calculations[dst_name + '_non_empty'] = False
                                            first_src = True
                                            for src in mp.source_entities:
                                                src_name = src.name
                                                # prepare the information
                                                src_idx, dst_idx, seq = f_.get('src_' + src_name +
                                                                               '_to_' + dst_name), \
                                                                        f_.get('dst_' + src_name +
                                                                               '_to_' + dst_name), \
                                                                        f_.get('seq_' + src_name +
                                                                               '_to_' + dst_name)

                                                # Transform the dimensions of the indices to the appropriate 2d size
                                                src_idx = tf.squeeze(src_idx)
                                                dst_idx = tf.squeeze(dst_idx)
                                                seq = tf.squeeze(seq)
                                                src_idx = tf.reshape(src_idx,
                                                                     [tf.cast(tf.size(src_idx), dtype=tf.int64)])
                                                dst_idx = tf.reshape(dst_idx,
                                                                     [tf.cast(tf.size(dst_idx), dtype=tf.int64)])
                                                seq = tf.reshape(seq, [tf.cast(tf.size(seq), dtype=tf.int64)])

                                                with tf.name_scope(src_name + '_to_' + dst_name) as _:
                                                    src_states = get_global_variable(self.calculations, str(src_name))

                                                    with tf.name_scope(
                                                            'create_message_' + src_name + '_to_' + dst_name) as _:
                                                        # check here if the indices is not empty??
                                                        self.src_messages = tf.gather(src_states, src_idx)
                                                        self.dst_messages = tf.gather(dst_states, dst_idx)
                                                        message_creation_models = src.message_formation

                                                        # by default, the source hs are the messages
                                                        result = self.src_messages
                                                        counter = 0

                                                        for op in message_creation_models:
                                                            if op is not None:  # if it is not direct_assignation
                                                                type_operation = op.type

                                                                if type_operation == 'neural_network':
                                                                    with tf.name_scope('apply_nn_' + str(counter)) as _:
                                                                        # careful. This name could overlap
                                                                        # with another model
                                                                        var_name = src_name + "_to_" + dst_name + \
                                                                                   '_message_creation_' + str(counter)
                                                                        message_creator = get_global_variable(
                                                                            self.calculations, var_name)
                                                                        result = op.apply_nn_msg(message_creator,
                                                                                                 self.calculations, f_,
                                                                                                 self.src_messages,
                                                                                                 self.dst_messages)

                                                                elif type_operation == 'product':
                                                                    with tf.name_scope(
                                                                            'apply_product_' + str(counter)) as _:
                                                                        product_input1 = \
                                                                            self.treat_message_function_input(
                                                                                op.input[0], f_)

                                                                        product_input2 = \
                                                                            self.treat_message_function_input(
                                                                                op.input[1], f_)
                                                                        result = op.calculate(product_input1,
                                                                                              product_input2)

                                                                if op.output_name is not None:
                                                                    save_global_variable(self.calculations,
                                                                                         op.output_name, result)
                                                            final_messages = result
                                                            counter += 1

                                                        # PREPARE FOR THE AGGREGATION
                                                        with tf.name_scope(
                                                                'combine_messages_' + src_name +
                                                                '_to_' + dst_name) as _:

                                                            ids = tf.stack([dst_idx, seq], axis=1)

                                                            lens = tf.math.unsorted_segment_sum(tf.ones_like(dst_idx),
                                                                                                dst_idx, num_dst)

                                                            # only a few aggregations actually needed to keep the order

                                                            max_len = tf.math.maximum(tf.cast(0, tf.int64),
                                                                                      tf.reduce_max(
                                                                                          seq) + 1)
                                                            # fix an error in the case that it is empty

                                                            message_dim = int(get_global_variable(self.calculations,
                                                                                                  "final_message_dim_"
                                                                                                  + str(idx_stage) +
                                                                                                  '_' + str(idx_msg)))

                                                            shape = tf.stack([num_dst, max_len, message_dim])
                                                            s = tf.scatter_nd(ids, final_messages,
                                                                              shape)
                                                            # find the input ordering it by sequence

                                                            aggr = mp.aggregations
                                                            if isinstance(aggr, ConcatAggr):
                                                                with tf.name_scope("concat_" + src_name) as _:
                                                                    if first_src:
                                                                        src_input = s
                                                                        final_len = lens
                                                                        first_src = False
                                                                    else:
                                                                        src_input = tf.concat([src_input, s],
                                                                                              axis=aggr.concat_axis)
                                                                        if aggr.concat_axis == 1:  # if axis=2, then
                                                                            # the number of messages received is the
                                                                            # same. Simply create bigger messages
                                                                            final_len += lens

                                                            elif isinstance(aggr, InterleaveAggr):
                                                                with tf.name_scope('add_' + src_name) as _:
                                                                    indices_source = f_.get(
                                                                        "indices_" + src_name + '_to_' + dst_name)
                                                                    if first_src:
                                                                        first_src = False
                                                                        src_input = s  # destinations x
                                                                        # max_of_sources_to_dest x dim_source
                                                                        indices = indices_source
                                                                        final_len = lens
                                                                    else:
                                                                        # destinations x
                                                                        # max_of_sources_to_dest_concat x dim_source
                                                                        src_input = tf.concat([src_input, s], axis=1)
                                                                        indices = tf.stack([indices, indices_source],
                                                                                           axis=0)
                                                                        final_len = tf.math.add(final_len, lens)

                                                            # if we must aggregate them together into a single
                                                            # embedding (sum, attention, edge_attention, ordered) the
                                                            # pipeline will either use the operations below or from
                                                            # above.
                                                            else:
                                                                # obtain the overall input of each of the destinations
                                                                if first_src:
                                                                    first_src = False
                                                                    src_input = s  # destinations x sources_to_dest x
                                                                    # dim_source
                                                                    comb_src_states, comb_dst_idx, comb_seq = \
                                                                        final_messages, dst_idx, seq  # we need this
                                                                    # for the attention and convolutional mechanism
                                                                    final_len = lens

                                                                else:
                                                                    # destinations x max_of_sources_to_dest_concat x
                                                                    # dim_source
                                                                    src_input = tf.concat([src_input, s], axis=1)
                                                                    comb_src_states = tf.concat(
                                                                        [comb_src_states, final_messages],
                                                                        axis=0)
                                                                    comb_dst_idx = tf.concat([comb_dst_idx, dst_idx],
                                                                                             axis=0)

                                                                    aux_lens = tf.gather(final_len,
                                                                                         dst_idx)  # lens of each
                                                                    # src-dst value
                                                                    aux_seq = seq + aux_lens  # sum to the sequences
                                                                    # the current length for each dest
                                                                    comb_seq = tf.concat([comb_seq, aux_seq], axis=0)

                                                                    final_len = tf.math.add(final_len, lens)

                                                aux = tf.cond(tf.size(src_idx) == 0, lambda: False, lambda: True)
                                                self.calculations[dst_name + '_non_empty'] = tf.math.logical_or(
                                                    self.calculations[dst_name + '_non_empty'], aux)

                                        # --------------
                                        # perform the actual aggregation
                                        aggrs = mp.aggregations

                                        # if ordered, we dont need to do anything. Already in the right shape
                                        # It only makes sense to do a pipeline with sum/attention/edge... operations??
                                        with tf.name_scope('aggregation') as _:
                                            for aggr in aggrs:
                                                with tf.name_scope(aggr.type) as _:
                                                    if aggr.type == 'sum':
                                                        src_input = aggr.calculate_input(comb_src_states,
                                                                                         comb_dst_idx,
                                                                                         num_dst)

                                                    elif aggr.type == 'mean':
                                                        src_input = aggr.calculate_input(comb_src_states,
                                                                                         comb_dst_idx,
                                                                                         num_dst)

                                                    elif aggr.type == 'min':
                                                        src_input = aggr.calculate_input(comb_src_states,
                                                                                         comb_dst_idx,
                                                                                         num_dst)

                                                    elif aggr.type == 'max':
                                                        src_input = aggr.calculate_input(comb_src_states,
                                                                                         comb_dst_idx,
                                                                                         num_dst)

                                                    elif aggr.type == 'std':
                                                        src_input = aggr.calculate_input(comb_src_states,
                                                                                         comb_dst_idx,
                                                                                         num_dst)

                                                    elif aggr.type == 'attention':
                                                        src_input = aggr.calculate_input(comb_src_states,
                                                                                         comb_dst_idx,
                                                                                         dst_states,
                                                                                         comb_seq, num_dst,
                                                                                         self.node_kernel,
                                                                                         self.attn_kernel)

                                                    elif aggr.type == 'edge_attention':
                                                        var_name = 'edge_attention_' + src_name + '_to_' + dst_name
                                                        edge_att_model = get_global_variable(self.calculations,
                                                                                             var_name)
                                                        comb_dst_states = tf.gather(dst_states,
                                                                                    comb_dst_idx)  # the destination
                                                        # state of each adjacency
                                                        model_input = tf.concat([comb_src_states, comb_dst_states],
                                                                                axis=1)

                                                        # define the shape of the input
                                                        dimension = self.dimensions[src_name] + self.dimensions[
                                                            dst_name]
                                                        model_input = tf.ensure_shape(model_input,
                                                                                      [None, dimension])

                                                        weights = edge_att_model(model_input)
                                                        src_input = aggr.calculate_input(comb_src_states,
                                                                                         comb_dst_idx,
                                                                                         num_dst,
                                                                                         weights)

                                                    # convolutional aggregation (the messages sent by the destination
                                                    # must have the same shape as the destinations)
                                                    elif aggr.type == 'convolution':
                                                        src_input = aggr.calculate_input(comb_src_states,
                                                                                         comb_dst_idx,
                                                                                         dst_states,
                                                                                         num_dst, self.conv_kernel)

                                                    elif aggr.type == 'interleave':
                                                        src_input = aggr.calculate_input(src_input, indices)

                                                    elif aggr.type == 'neural_network':
                                                        # concatenate in the axis 0 all the input tensors
                                                        var_name = 'aggr_nn'
                                                        aggregator_nn = get_global_variable(self.calculations,
                                                                                            var_name)
                                                        src_input = aggr.apply_nn(aggregator_nn, self.calculations,
                                                                                  f_)

                                                # save the result of this operation with its output_name
                                                if aggr.output_name is not None:
                                                    save_global_variable(self.calculations, aggr.output_name,
                                                                         src_input)

                                            # this is the final one that passes to the update
                                            # save the src_input used for the update
                                            save_global_variable(self.calculations, 'update_lens_' + dst_name,
                                                                 final_len)
                                            save_global_variable(self.calculations, 'update_input_' + dst_name,
                                                                 src_input)

                                # ---------------------------------------
                                # updates
                                with tf.name_scope('updates') as _:
                                    for mp in stage[1]:
                                        dst_name = mp.destination_entity

                                        with tf.name_scope('update_' + dst_name) as _:
                                            if self.calculations[dst_name + '_non_empty']:
                                                update_model = mp.update
                                                src_input = get_global_variable(self.calculations,
                                                                                'update_input_' + dst_name)
                                                old_state = get_global_variable(self.calculations, dst_name)

                                                # if there was no accumulated input (no adjacencies)
                                                if update_model is None:
                                                    # by default use the aggregated messages as new state This should
                                                    # only be compatible with sum/attention/convolution (obtain a
                                                    # single tensor)
                                                    new_state = src_input

                                                # recurrent update
                                                elif isinstance(update_model, RNNOperation):
                                                    model = get_global_variable(self.calculations, dst_name + '_update')
                                                    if not mp.aggregations_global_type:
                                                        # should this be the source dimensions??? CHECK
                                                        dst_dim = int(self.dimensions[dst_name])
                                                        new_state = \
                                                            update_model.model.perform_unsorted_update(model,
                                                                                                       src_input,
                                                                                                       old_state,
                                                                                                       dst_dim)

                                                    # if the aggregation was ordered or concat
                                                    else:
                                                        final_len = get_global_variable(self.calculations,
                                                                                        'update_lens_' + dst_name)
                                                        new_state = update_model.model.perform_sorted_update(model,
                                                                                                             src_input,
                                                                                                             dst_name,
                                                                                                             old_state,
                                                                                                             final_len)

                                                # feed-forward update:
                                                # restriction: It can only be used if the aggreagation was not ordered.
                                                else:
                                                    var_name = dst_name + "_ff_update"
                                                    update = get_global_variable(self.calculations, var_name)

                                                    # now we need to obtain for each adjacency the concatenation of
                                                    # the source and the destination
                                                    update_input = tf.concat([src_input, old_state], axis=1)
                                                    new_state = update(update_input)

                                                # update the old state
                                                self.calculations[dst_name] = new_state

            # -----------------------------------------------------------------------------------
            # READOUT PHASE
            with tf.name_scope('readout_predictions') as _:
                readout_operations = self.model_info.get_readout_operations()
                counter = 0
                n = len(readout_operations)
                for j in range(n):
                    operation = readout_operations[j]
                    with tf.name_scope(operation.type) as _:
                        if operation.type == 'neural_network':
                            var_name = 'readout_model_' + str(counter)
                            readout_nn = get_global_variable(self.calculations, var_name)
                            result = operation.apply_nn(readout_nn, self.calculations, f_)

                        elif operation.type == "pooling":
                            # obtain the input of the pooling operation
                            first = True
                            for input_name in operation.input:
                                aux = get_global_var_or_input(self.calculations, input_name, f_)
                                if first:
                                    pooling_input = aux
                                    first = False
                                else:
                                    pooling_input = tf.concat([pooling_input, aux], axis=0)

                            result = operation.calculate(pooling_input)

                        elif operation.type == 'product':
                            product_input1 = get_global_var_or_input(self.calculations, operation.input[0], f_)
                            product_input2 = get_global_var_or_input(self.calculations, operation.input[1], f_)
                            result = operation.calculate(product_input1, product_input2)

                        # extends the two inputs following the adjacency list that connects them both. CHECK!!
                        elif operation.type == 'extend_adjacencies':  # CHECK!!!!!
                            adj_src = f_.get('src_' + operation.adj_list)
                            adj_dst = f_.get('dst_' + operation.adj_list)

                            src_states = get_global_var_or_input(self.calculations, operation.input[0], f_)
                            dst_states = get_global_var_or_input(self.calculations, operation.input[1], f_)

                            extended_src, extended_dst = operation.calculate(src_states, adj_src, dst_states, adj_dst)
                            save_global_variable(self.calculations, operation.output_name[0], extended_src)
                            save_global_variable(self.calculations, operation.output_name[1], extended_dst)

                        elif operation.type == 'concat':
                            inputs = []
                            for param in operation.input:
                                inputs.append(get_global_var_or_input(self.calculations, param, f_))
                            result = operation.calculate(inputs)
                        # output of the readout
                        if operation.type != 'extend_adjacencies':
                            if j == n - 1:  # last one
                                return result
                            else:
                                save_global_variable(self.calculations, operation.output_name, result)

                    counter += 1

    def treat_message_function_input(self, var_name, f_):
        if var_name == 'source':
            new_input = self.src_messages
        elif var_name == 'destination':
            new_input = self.dst_messages
        else:
            new_input = get_global_var_or_input(self.calculations, var_name, f_)
        return new_input
