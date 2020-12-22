import tensorflow as tf
import tensorflow.keras.activations
from keras import backend as K
import sys
from ignnition.utils import *
from ignnition.model_classes import *


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
        self.output_label = op.get('output_label', None)
        self.output_name = op.get('output_name', None)
        self.input = op.get('input', None)

    def find_total_input_dim(self, dimensions, calculations):
        if self.input is not None:
            input_nn = self.input
            input_dim = 0
            for i in input_nn:
                if '_initial_state' in i:
                    i = i.split('_initial_state')[0]
                if i in dimensions:
                    dimension = dimensions[i]
                else:
                    dimension = calculations[i + '_dim']# take the dimension from here or from self.dimensions
                input_dim += dimension
            return input_dim

    def obtain_total_input_dim_message(self, dimensions, calculations, dst_name, src):
        # Find out the dimension of the model
        input_nn = self.input
        input_dim = 0
        for i in input_nn:
            if i == 'source':
                input_dim += int(dimensions.get(src.name))
            elif i == 'destination':
                input_dim += int(dimensions.get(dst_name))
            else:
                dimension = calculations[i + '_dim']
                input_dim += dimension
        return input_dim

    def compute_all_input(self, calculations, f_):
        first = True
        for i in self.input:
            if '_initial_state' in i:
                i = i.split('_initial_state')[0]

            new_input = get_global_var_or_input(calculations, i, f_)

            if len(tf.shape(new_input)) == 1:
                new_input = tf.expand_dims(new_input, axis=-1)

            # accumulate the results
            if first:
                first = False
                input_nn = new_input
            else:
                input_nn = tf.concat([input_nn, new_input], axis=1)

        return input_nn

    def compute_all_input_msg(self, calculations, f_, src_msgs, dst_msgs):
        first = True
        for i in self.input:
            if i == 'source':
                new_input = src_msgs
            elif i == 'destination':
                new_input = dst_msgs
            else:
                new_input = get_global_var_or_input(calculations, i, f_)

            if len(tf.shape(new_input)) == 1:
                new_input = tf.expand_dims(new_input, axis=-1)

            # accumulate the results
            if first:
                first = False
                input_nn = new_input
            else:
                input_nn = tf.concat([input_nn, new_input], axis=1)

        return input_nn


class Build_state(Operation):
    """
        Subclass of Readout_operation that represents the product operation

        Attributes:
        ----------
        type_product:    str
            Type of product to be used
        output_name:    int
            Name to save the output of the operation with

        """

    def __init__(self, op, entity_name, entity_dim):
        super(Build_state, self).__init__(op)
        self.entity_name = entity_name
        self.entity_dim = entity_dim

    def calculate_hs(self, calculations, f_):
        state = self.compute_all_input(calculations, f_)

        remaining_zeros = tf.cast(self.entity_dim - tf.shape(state)[1], tf.int64)

        shape = tf.stack([tf.cast(f_.get('num_' + self.entity_name), tf.int64), remaining_zeros], axis=0)  # shape (2,)
        state = tf.concat([state, tf.zeros(shape)], axis=1)
        return state

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

            result = tf.cast(result, tf.float32)

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

    def apply_nn(self, model, calculations, f_, readout=False):
        input_nn = self.compute_all_input(calculations, f_)

        if readout and len(tf.shape(input_nn)) == 1:
                input_nn = tf.expand_dims(input_nn, axis=0)

        with tf.name_scope('pass_to_nn') as _:
            return model(input_nn)

    def apply_nn_msg(self, model, calculations, f_, src_msgs, dst_msgs):
        input_nn = self.compute_all_input_msg(calculations, f_, src_msgs, dst_msgs)

        with tf.name_scope('pass_to_nn') as _:
            return model(input_nn)

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
