import tensorflow as tf
import tensorflow.keras.activations
from keras import backend as K
import sys
from ignnition.utils import *
from ignnition.operation_classes import *

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
