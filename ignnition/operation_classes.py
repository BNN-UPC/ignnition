import sys

import tensorflow as tf

from ignnition.model_classes import FeedForwardModel, Recurrent_Update_Cell
from ignnition.utils import print_failure
from ignnition.utils import get_global_var_or_input


class Operation:
    """
    Class that represents a general operation

    Attributes:
    ----------
    layer_type:    str
        Type of operation (identified by a keyword)
    output_name:    str
        Name to save the output of the operation with (if any)
    output_label:    str
        Name to save the output label which we aim to predict (only needed for the last one)
    input:    array
        Array of the input names

    Methods:
    ----------
    find_total_input_dim(self, dimensions, calculations)
        Computes the total dimension of all the inputs of this operation
    obtain_total_input_dim_message(self, dimensions, calculations, dst_name, src)
        Computes the total dimension of all the inputs of this operation (includes source and destination
        keywords which are useful to define an operation for the message creation function)
    compute_all_input(self, calculations, f_)
        Given the array of names, it computes the final input of th operation by concatenating them together.
    compute_all_input_msg(self, calculations, f_, src_msgs, dst_msgs)
        Same as the previous but treating the source and destination keywords accordingly
    """

    def __init__(self, op):
        """
        Parameters
        ----------
        op:    dict
            Dictionary with the data defining this general operation
        """

        self.type = op.get('type')
        self.output_name = op.get('output_name', None)

        self.output_label = op.get('output_label', None)

        if self.output_label is not None:
            # There may be more than one output_label
            self.output_label = [output.split('$')[-1] for output in
                                 self.output_label]  # delete the $ from the output label

        # parse the input of the operation
        self.input = []
        self.source_dataset = False
        self.destination_dataset = False
        if 'input' in op:
            for input_item in op.get('input'):
                if '$source' == input_item or '$destination' == input_item:
                    print_failure(
                        'The keywords source and destination are reserved keywords. Thus, they cannot name feature '
                        'from the dataset. Check that you really meant to use $, indicating that its a feature '
                        'from the dataset')
                else:
                    self.input.append(input_item.split('$')[-1])  # delete the $ from the inputs (if any)

    def find_total_input_dim(self, dimensions, calculations):
        """
        Parameters
        ----------
        dimensions:    dict
           Dictionary with the dimensions of each tensor (indexed by name)
        calculations:    dict
           Dictionary with the current calculations throughout the execution of the GNN model
        """
        if self.input is not None:
            input_nn = self.input
            input_dim = 0
            dimension = None
            for i in input_nn:
                if '_initial_state' in i:
                    i = i.split('_initial_state')[0]

                if i in dimensions:
                    dimension = dimensions[i]
                elif i + '_dim' in calculations:
                    dimension = calculations[i + '_dim']  # take the dimension from here or from self.dimensions
                else:
                    print_failure("Keyword " + i + " used in the model definition was not recognized")

                input_dim += dimension
            return input_dim

    def obtain_total_input_dim_message(self, dimensions, calculations, dst_name, src):
        """
        Parameters
        ----------
        dimensions:    dict
           Dictionary with the dimensions of each tensor (indexed by name)
        calculations:    dict
           Dictionary with the current calculations throughout the execution of the GNN model
        dst_name: str
            Name of the destination entity
        src: Source_mp object
            Object that includes the information about the source entity of the mp
        """
        # Find out the dimension of the model
        input_nn = self.input
        input_dim = 0
        for i in input_nn:
            if i == 'source':
                input_dim += int(dimensions.get(src.name))
            elif i == 'destination':
                input_dim += int(dimensions.get(dst_name))
            elif i in dimensions:
                input_dim += int(dimensions[i])
            elif i + '_dim' in calculations:
                input_dim += dimensions
            else:
                print_failure("Keyword " + i + " used in the message passing was not recognized.")

        return input_dim

    def compute_all_input(self, calculations, f_):
        """
        Parameters
        ----------
        calculations:    dict
           Dictionary with the dimensions of each tensor (indexed by name)
        f_:    dict
           Dictionary with the data of the current sample
        """

        first = True
        for i in self.input:
            if '_initial_state' in i:
                i = i.split('_initial_state')[0]

            new_input = get_global_var_or_input(calculations, i, f_)

            # accumulate the results
            if first:
                first = False
                input_nn = new_input
            else:
                input_nn = tf.concat([input_nn, new_input], axis=1)
        return input_nn

    def compute_all_input_msg(self, calculations, f_, src_msgs, dst_msgs):
        """
        Parameters
        ----------
        calculations:    dict
            Dictionary with the current calculations throughout the execution of the GNN model
        f_:    dict
            Dictionary with the data of the current sample
        src_msgs:
            Tensor with all the messages of the source nodes for each of the edges
        src_msgs:
            Tensor with all the messages of the destination nodes for each of the edges
        """

        first = True
        for i in self.input:
            if i == 'source':
                new_input = src_msgs
            elif i == 'destination':
                new_input = dst_msgs
            else:
                new_input = get_global_var_or_input(calculations, i, f_)

            # ensure that this tensor is 2-D
            new_input = tf.reshape(new_input, [-1] + [tf.shape(new_input)[-1]])

            # accumulate the results
            if first:
                first = False
                input_nn = new_input
            else:
                new_input = tf.cast(new_input, dtype=tf.float32)
                input_nn = tf.concat([input_nn, new_input], axis=1)
        return input_nn


class BuildState(Operation):
    """
    Subclass of Operation that represents the operation of building the hs of a given entity layer_type

    Attributes:
    ----------
    entity_name:    str
        Name of the treated entity
    entity_dim:    int
        Maximum dimension of the entity's hs. If this dimension is not met, we pad it with 0s.

    Methods:
    ----------
    calculate_hs(self, calculations, f_)
        Computes the hidden states for a given entity
    """

    # TODO: Error message if we pass the maximum dimension.
    def __init__(self, op, entity_name, entity_dim):
        """
        Parameters
        ----------
        op:    dict
            Dictionary with the data defining this general operation
        entity_name: str
            Name of the entity which we aim to build the state of
        entity_dim: int
            Maximum dimension of the entity's hs. If this dimension is not met, we pad it with 0s.
        """

        super(BuildState, self).__init__(op)
        self.entity_name = entity_name
        self.entity_dim = entity_dim

    def calculate_hs(self, calculations, f_):
        """
        Parameters
        ----------
        calculations:    dict
            Dictionary with the current calculations throughout the execution of the GNN model
        f_:    dict
            Dictionary with the data of the current sample
        """

        if self.input:
            state = self.compute_all_input(calculations, f_)
            remaining_zeros = tf.cast(self.entity_dim - tf.shape(state)[1], tf.int64)
            shape = tf.stack([tf.cast(f_.get('num_' + self.entity_name), tf.int64), remaining_zeros],
                             axis=0)  # shape (2,)
            state = tf.concat([state, tf.zeros(shape)], axis=1)
        else:
            shape = tf.stack([tf.cast(f_.get('num_' + self.entity_name), tf.int64), self.entity_dim],
                             axis=0)  # shape (2,)
            state = tf.zeros(shape)
        return state


class ProductOperation(Operation):
    """
    Subclass of Operation class that represents the product operation between two tensors (also considers several
    types of products)

    Attributes:
    ----------
    type_product:    str
        Type of product to be used
    output_name:    str
        Name to save the output of the operation with (included in the super object)

    Methods:
    ----------
    calculate(self, product_input1, product_input2)
        Computes the product specified by the user of the two input tensors.
    """

    def __init__(self, op):
        """
        Parameters
        ----------
        op:    dict
            Dictionary with the data defining this product operation
        """

        super(ProductOperation, self).__init__(op)
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
                result = tf.tensordot(product_input1, product_input2, axes=[[1], [1]])

                # the correct values are in the diagonal (IMPROVE THIS)
                # This does the dot product row by row (so independently for each adjacency)
                result = tf.linalg.tensor_diag_part(result)
                result = tf.expand_dims(result, axis=-1)

            elif self.type_product == 'element_wise':
                result = tf.math.multiply(product_input1, product_input2)

            elif self.type_product == 'mat_mult':
                result = tf.tensordot(product_input1, product_input2, axes=[[2], [1]])
                result = tf.squeeze(result, axis=2)

            result = tf.cast(result, tf.float32)

            return result

        except:
            print_failure(
                'The product operation between ' + product_input1 + ' and ' + product_input2 +
                ' failed. Check that the dimensions are compatible.')


class PoolingOperation(Operation):
    """
    Subclass of Operation class that represents the pooling operation (which given an array of tensors, it computes
    a global representation of them).

    Attributes:
    ----------
    type_pooling:    str
        Type of pooling to be used

    Methods:
    --------
    calculate(self, pooling_input)
        Applies the pooling operation specified by the user to an input
    """

    def __init__(self, operation):
        """
        Parameters
        ----------
        operation:    dict
            Dictionary with the readout_model parameters
        """

        super(PoolingOperation, self).__init__(operation)
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
            result = tf.reshape(result, [-1] + [tf.shape(result)[0]])

        elif self.type_pooling == 'mean':
            result = tf.reduce_mean(pooling_input, 0)
            result = tf.reshape(result, [-1] + [tf.shape(result)[0]])

        elif self.type_pooling == 'max':
            result = tf.reduce_max(pooling_input, 0)
            result = tf.reshape(result, [-1] + [tf.shape(result)[0]])

        return result


class FeedForwardOperation(Operation):
    """
    Subclass of Operation that represents a NN operation which consists of passing a given input to the specified NN.

    Attributes:
    ----------
    model:    FeedForwardModel obj
        Object representing the NN.

    Methods:
    --------
    apply_nn(self, model, calculations, f_, readout=False)
        Applies the input of this operation to the specified NN. It computes itself the input of this op given the
        input sample.
    apply_nn_msg(self, model, calculations, f_, src_msgs, dst_msgs)
        Applies the input of this operation to the specified NN. It computes itself the input of this op given the
        input sample. It also takes into consideration the source and destination keywords used in the message creation
        function
    """

    def __init__(self, op, model_role):
        """
        Parameters
        ----------
        op:    dict
            Dictionary with the data defining this feed_forward operation
        model_role: str
            Defines the role of this operation (e.g., message_creation). Only useful for variable naming and debugging
        """

        super(FeedForwardOperation, self).__init__(op)

        # we need somehow to find the number of extra_parameters beforehand
        self.model = FeedForwardModel({'architecture': op.get('architecture')}, model_role=model_role)

    def apply_nn(self, model, calculations, f_):
        """
        Parameters
        ----------
        model: FeedForwardModel obj
            Object representing the NN.
        calculations:    dict
            Dictionary with the current calculations throughout the execution of the GNN model
        f_:    dict
            Dictionary with the data of the current sample
        """

        input_nn = self.compute_all_input(calculations, f_)

        input_size = model.input_shape[-1]
        input_nn = tf.ensure_shape(input_nn, [None, input_size])
        return model(input_nn)

    def apply_nn_msg(self, model, calculations, f_, src_msgs, dst_msgs):
        """
        Parameters
        ----------
        model: FeedForwardModel obj
            Object representing the NN.
        calculations:    dict
            Dictionary with the current calculations throughout the execution of the GNN model
        f_:    dict
            Dictionary with the data of the current sample
        src_msgs: tensor
            Tensor with the input messages for each of the edges (source)
        dst_msgs: tensor
            Tensor with the input messages for each of the edges (destination)
        """
        input_nn = self.compute_all_input_msg(calculations, f_, src_msgs, dst_msgs)
        input_size = model.input_shape[-1]
        input_nn = tf.ensure_shape(input_nn, [None, input_size])
        return model(input_nn)


class RNNOperation(Operation):
    """
    Subclass of Operation that represents a RNN operation which consists of passing a given input to the specified RNN.

    Attributes:
    ----------
    model:    RNNOperation obj
        Object representing the NN.
    input: str
        Name of the input to be fed to this RNN.
   """

    def __init__(self, op):
        """
        Parameters
        ----------
        op:    dict
            Dictionary with the data defining this product operation
        """

        super(RNNOperation, self).__init__(op)

        # we need somehow to find the number of extra_parameters beforehand
        cell_architecture = op['architecture'][0]  # in this case only one layer will be specified.
        layer_type = cell_architecture['type_layer']
        self.model = Recurrent_Update_Cell(layer_type=layer_type, parameters=cell_architecture)
        self.input = op.get('input', None)


# TODO: check that it works
class ExtendAdjacencies(Operation):
    """
    Subclass of oPERATION that represents the extend_adjacencies operation

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
        """
        Parameters
        ----------
        op:    dict
            Dictionary with the data defining this product operation
        """

        super(ExtendAdjacencies, self).__init__({'type': op['type'], 'input': op['input']})
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
        except Exception:
            print_failure('Extending the adjacency list ' + str(self.adj_list) +
                          ' was not possible. Check that the indexes of the source of the adjacency '
                          'list match the input given.')

        try:
            extended_dst = tf.gather(dst_states, adj_dst)
        except Exception:
            print_failure('Extending the adjacency list ' + str(self.adj_list) +
                          ' was not possible. Check that the indexes of the destination of '
                          'the adjacency list match the input given.')

        return extended_src, extended_dst


class Concat(Operation):
    """
    Subclass of Operation class that represents the product operation between two tensors (also considers
    several types of products)

    Attributes:
    ----------
    type_product:    str
        Type of product to be used
    output_name:    str
        Name to save the output of the operation with (included in the super object)

    Methods:
    ----------
    calculate(self, product_input1, product_input2)
        Computes the product specified by the user of the two input tensors.
    """

    def __init__(self, op):
        """
        Parameters
        ----------
        op:    dict
            Dictionary with the data defining this product operation
        """

        super(Concat, self).__init__(op)
        self.axis = op.get('axis', 0)

    def calculate(self, inputs):
        """
        Parameters
        ----------
        inputs:    tensor
        """

        try:
            result = tf.concat(inputs, axis=self.axis)
            return result

        except Exception:
            print_failure(
                'The concat operation failed. Check that the dimensions are compatible.')
            sys.exit(1)
