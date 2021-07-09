import tensorflow as tf
import tensorflow.keras.backend as K

from ignnition.operation_classes import FeedForwardOperation


class Aggregation:
    """
    A class that represents a general aggregation operation

    Attributes
    ----------
    aggr_def:    dict
        Dictionary with the information of the aggregation function
    """

    def __init__(self, aggr_def):
        """
        Parameters
        ----------
        aggr_def:    dict
            Data corresponding to the general aggregation definition
        """
        self.type = aggr_def.get('type')
        self.output_name = aggr_def.get('output_name', None)


class SumAggr(Aggregation):
    """
    A subclass that represents the Sum aggreagtion operation (which sums all the input messages together for each of
    the destination nodes).

    Methods:
    ----------
    calculate_input(self, comb_src_states, comb_dst_idx, num_dst)
        Returns the sum of all the input messages for each of the destination nodes.
    """

    def __init__(self, aggr_def):
        """
        Parameters
        ----------
        aggr_def:    dict
            Data corresponding to the sum aggregation definition
        """
        super(SumAggr, self).__init__(aggr_def)

    def calculate_input(self, comb_src_states, comb_dst_idx, num_dst):
        """
        Parameters
        ----------
        comb_src_states:    tensor
           Indices of the source nodes for each of the adjacencies to consider.
        comb_dst_idx:    tensor
           Indices of the destination nodes for each of the adjacencies to consider.
        num_dst:    int
           Number of source entities
        """

        src_input = tf.math.unsorted_segment_sum(comb_src_states, comb_dst_idx, num_dst)
        return src_input


class MeanAggr(Aggregation):
    """
    A subclass that represents the mean aggregation operation

    Methods:
    ----------
    calculate_input(self, comb_src_states, comb_dst_idx, num_dst)
        Returns the mean of all the input messages for each of the destination nodes.
    """

    def __init__(self, aggr_def):
        """
        Parameters
        ----------
        aggr_def:    dict
            Data corresponding to the mean aggregation definition
        """
        super(MeanAggr, self).__init__(aggr_def)

    def calculate_input(self, comb_src_states, comb_dst_idx, num_dst):
        """
        Parameters
        ----------
        comb_src_states:    tensor
           Indices of the source nodes for each of the adjacencies to consider.
        comb_dst_idx:    tensor
           Indices of the destination nodes for each of the adjacencies to consider.
        num_dst:    int
           Number of source entities
        """

        neighbours_mean = tf.math.unsorted_segment_mean(comb_src_states, comb_dst_idx, num_dst)
        return neighbours_mean


class MaxAggr(Aggregation):
    """
    A subclass that represents the Max aggregation operation

    Methods:
    ----------
    calculate_input(self, comb_src_states, comb_dst_idx, num_dst)
        Returns the max of all the input messages for each of the destination nodes.
    """

    def __init__(self, aggr_def):
        """
        Parameters
        ----------
        aggr_def:    dict
            Data corresponding to the max aggregation definition
        """
        super(MaxAggr, self).__init__(aggr_def)

    def calculate_input(self, comb_src_states, comb_dst_idx, num_dst):
        """
        Parameters
        ----------
        comb_src_states:    tensor
           Indices of the source nodes for each of the adjacencies to consider.
        comb_dst_idx:    tensor
           Indices of the destination nodes for each of the adjacencies to consider.
        num_dst:    int
           Number of source entities
        """

        src_input = tf.math.unsorted_segment_max(comb_src_states, comb_dst_idx, num_dst)
        return src_input


class MinAggr(Aggregation):
    """
    A subclass that represents the Sum aggreagtion operation

    Methods:
    ----------
     calculate_input(self, comb_src_states, comb_dst_idx, num_dst)
        Returns the min of all the input messages for each of the destination nodes.
    """

    def __init__(self, aggr_def):
        """
        Parameters
        ----------
        aggr_def:    dict
            Data corresponding to the min aggregation definition
        """
        super(MinAggr, self).__init__(aggr_def)

    def calculate_input(self, comb_src_states, comb_dst_idx, num_dst):
        """
        Parameters
        ----------
        comb_src_states:    tensor
           Indices of the source nodes for each of the adjacencies to consider.
        comb_dst_idx:    tensor
           Indices of the destination nodes for each of the adjacencies to consider.
        num_dst:    int
           Number of source entities
        """

        src_input = tf.math.unsorted_segment_min(comb_src_states, comb_dst_idx, num_dst)
        return src_input


# toDO: Finish this operation
class StdAggr(Aggregation):
    """
    A subclass that represents the Std aggreagtion operation

    Methods:
    ----------
    calculate_input(self, comb_src_states, comb_dst_idx, num_dst)
        Returns the std of all the input messages for each of the destination nodes.
    """

    def __init__(self, aggr_def):
        """
        Parameters
        ----------
        aggr_def:    dict
            Data corresponding to the std aggregation definition
        """
        super(StdAggr, self).__init__(aggr_def)

    def calculate_input(self, comb_src_states, comb_dst_idx, num_dst):
        """
        Parameters
        ----------
        comb_src_states:    tensor
           Indices of the source nodes for each of the adjacencies to consider.
        comb_dst_idx:    tensor
           Indices of the destination nodes for each of the adjacencies to consider.
        num_dst:    int
           Number of source entities
        """

        src_input = tf.math.unsorted_segment_sum(comb_src_states, comb_dst_idx, num_dst)
        return src_input


class AttentionAggr(Aggregation):
    """
    A subclass that represents the attention aggregation operation

    Attributes
    ----------
    weight_initialization:    str
        Indicates how the weights are initialized (if any parameter is specified at all)

    Methods:
    ----------
    calculate_input(self, comb_src_states, comb_dst_idx, dst_states, comb_seq, num_dst, node_kernel, attn_kernel)
        Computes the attention mechanism of all the input messages for each destination node. This aggregation
        corresponds to the one proposed for Graph Attention Networks.
    """

    def __init__(self, aggr_def):
        """
        Parameters
        ----------
        aggr_def:    dict
            Data corresponding to the attention aggregation definition
        """
        super(AttentionAggr, self).__init__(aggr_def)
        self.weight_initialization = aggr_def.get('weight_initialization', None)

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
            Indices that indicate the sequences for each destination node.
        num_dst:    int
            Number of destination entity nodes
        node_kernel:    tf object
            node_kernel object to transform the source's and destination's hs shape
        attn_kernel:    tf.object
            Attn_kernel object
        """

        # obtain the source states  (NxF1)
        h_src = tf.identity(comb_src_states)

        # dst_states <- (N x F2)
        # F2 = int(self.dimensions[mp.destination_entity])

        # new number of features (right now set to F1, but could be different)
        # F_ = F1

        # node_kernel = F1 x F1 (we could change the output dimension)
        # transformed_states_sources = NxF1 X F1xF1 = NxF1
        transformed_states_sources = K.dot(h_src, node_kernel)  # (W h_i for every source)

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
        coef = tf.keras.activations.softmax(aux, axis=0)

        # sum them all together using the coefficients (average)
        final_coef = tf.gather_nd(coef, ids)
        weighted_inputs = comb_src_states * final_coef

        src_input = tf.math.unsorted_segment_sum(weighted_inputs, comb_dst_idx,
                                                 num_dst)
        return src_input


class EdgeAttentionAggr(Aggregation):
    """
    A subclass that represents the Edge attention aggregation operation

    Attributes
    ----------
    aggr_model:    str
        Feed forward model used to compute the weights for each of the edges.

    Methods:
    ----------
    get_model()
        Return the aggregation model

    calculate_input(self, comb_src_states, comb_dst_idx, num_dst, weights)
        Computes the edge attention, based on computing a weight for each input message by passing its source
        and destination hs to a NN.
    """

    def __init__(self, op):
        """
        Parameters
        ----------
        op:    dict
            Dictionary with the user's definition of this operation
        """

        super(EdgeAttentionAggr, self).__init__(op)
        del op['type']
        self.aggr_model = FeedForwardOperation(op, model_role='edge_attention')

    def get_model(self):
        return self.aggr_model.model

    def calculate_input(self, comb_src_states, comb_dst_idx, num_dst, weights):
        """
        Parameters
        ----------
        comb_src_states:    tensor
            Source hs
        comb_dst_idx:   tensor
            Destination indexes to be combined with (src -> dst)
        num_dst:    int
            Number of destination entity nodes
        weights:    tensor
            This are the weights for each of the adjacencies to be applied to the input messages
        """

        # apply the attention mechanism
        weighted_inputs = weights * comb_src_states
        # sum by destination nodes
        src_input = tf.math.unsorted_segment_sum(weighted_inputs, comb_dst_idx, int(num_dst))
        return src_input


class ConvAggr(Aggregation):
    """
    A subclass that represents the Convolution aggregation operation

    Attributes
    ----------
    activation_function: str
        Name of the activation function to be used (if any)
    weight_initialization:    str
        Indicates how the weights are initialized (if any parameter is specified at all)

    Methods:
    ----------
    calculate_input(self, comb_src_states, comb_dst_idx, dst_states, num_dst, kernel)
        Calculates the result of applying the convolution mechanism (proposed for the graph convolutional NN)
    """

    def __init__(self, attr):
        """
        Parameters
        ----------
        attr:    dict
            Data corresponding to the convolutional aggregation definition
        """
        super(ConvAggr, self).__init__(attr)
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
            Kernel object to transform the source's hs shape
        """

        # MATHEMATICAL FORMULATION:
        # CONVOLUTION: h_i^t = SIGMA(SUM_N(i) (1 / (sqrt(deg(i)) * sqrt(deg(j))) * w * x_j^(t-1))
        # = h_i^t = SIGMA( 1 / sqrt(deg(i)) * SUM_N(i) (1 / (sqrt(deg(j))) * w * x_j^(t-1))
        # implemented: h_i^t = SIGMA(1 / sqrt(deg(i)) * SUM_N(i) w * x_j^(t-1))

        # comb_src_states = N x F    kernel = F x F
        weighted_input = tf.linalg.matmul(comb_src_states, kernel)

        # normalize each input dividing by sqrt(deg(j)) (only applies if they are from the same entity as
        # the destination node)??

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

        # normalize all the values dividing by sqrt(dst_deg)
        normalized_val = tf.math.divide_no_nan(total_sum, dst_deg)

        # normalize by mean and variance  (CHECK) This is the node normalization
        mean = tf.math.reduce_mean(normalized_val)
        var = tf.math.reduce_std(normalized_val)
        normalized_val = (normalized_val - mean) / var

        # apply the non-linearity
        activation_func = getattr(tf.nn, self.activation_function)

        return activation_func(normalized_val)


class InterleaveAggr(Aggregation):
    """
    A subclass that represents the Interleave aggregation operation

    Attributes
    ----------
    combination_definition:    str
        Defines the name from the dataset with this custom definition

    Methods:
    ----------
    calculate_input(self, src_input, indices)
        Computes the result of applying the interleave mechanism. This mechanism takes as input a tensor with all the
        input messages of several sources to a same destination entity layer_type, and creates a custom array of the
        input messages. With these, for instance, a destination node can receive the a tensor where the pair messages
        are from an entity layer_type and the odds from the other.
    """

    def __init__(self, aggr_def):
        """
        Parameters
        ----------
        aggr_def:    dict
            Data corresponding to the interleave aggregation definition
        """

        super(InterleaveAggr, self).__init__(aggr_def)
        self.combination_definition = aggr_def.get('interleave_definition')

    def calculate_input(self, src_input, indices):
        """
        Parameters
        ----------
        src_input:    tensor
            Combined sources hs
        indices:    tensor
            Indices to reorder for the interleave
        """

        # destinations x max_of_sources_to_dest_concat x dim_source ->  (max_of_sources_to_dest_concat x
        # destinations x dim_source)
        src_input = tf.transpose(src_input, perm=[1, 0, 2])
        indices = tf.reshape(indices, [-1, 1])

        src_input = tf.scatter_nd(indices, src_input,
                                  tf.shape(src_input, out_type=tf.int64))

        # (max_of_sources_to_dest_concat x destinations x dim_source) -> destinations x max_of_sources_to_dest_concat
        # x dim_source
        src_input = tf.transpose(src_input, perm=[1, 0, 2])
        return src_input


class ConcatAggr(Aggregation):
    """
    A subclass that represents the Concat aggregation operation

    Attributes
    ----------
    concat_axis:    str
        Axis to concatenate the input
    """

    def __init__(self, attr):
        """
        Parameters
        ----------
        attr:    dict
            Data corresponding to the concat aggregation definition
        """
        super(ConcatAggr, self).__init__(attr)
        self.concat_axis = int(attr.get('concat_axis'))
