import ignnition
import tensorflow as tf
from pathlib import Path

noise_power = tf.constant(4e-12)


@tf.function()
def compute_sum_rate(power, loss, weights, N):
    """Compute Sum Rate from power allocation and channle loss matrix."""
    # Prepare power tensor
    power_tiled = tf.tile(power, [N, 1])
    rx_power = tf.math.square(tf.multiply(loss, power_tiled))
    # Prepare masks diagonal and off-diagonal masks
    mask = tf.eye(N)
    mask_inverse = tf.ones_like(mask) - mask
    # Compute valid power/interferences per transciever-reciever-pair
    valid_rx_power = tf.reduce_sum(tf.multiply(rx_power, mask), axis=-1)
    interference = tf.reduce_sum(tf.multiply(rx_power, mask_inverse), axis=-1)
    interference += tf.repeat(noise_power, tf.shape(interference))
    # Compute SINR rates
    sinr = tf.ones(N) + tf.divide(valid_rx_power, interference)
    sum_rate = tf.divide(tf.math.log(sinr), tf.math.log(tf.constant(2, dtype=tf.float32)))
    weighted_sum_rate = tf.multiply(weights, sum_rate)
    return tf.reduce_mean(tf.reduce_sum(weighted_sum_rate, -1))


@tf.function()
def sum_rate_loss(y_true, y_pred):
    """SINR sum rate loss.

    Loss function for Radio Resource Management example, computing the expected sum rate value.
    Inputs are batched tensors with shape (b, n, ?) where b is batch_size, n is the number of
    nodes in the graph an ? changes depending on input.

    Parameters
    ----------
    y_true : tf.Tensor
        Batched tensor with with shape (b, n, n) containing for each node the path losses from the
        node (transceiver-reciever-pair) to all others, include itself.
    y_pred : tf.Tensor
        Batched tensor with GNN output, containing a hidden state with shape (b, n, 4), where second
        last element is the allocated power to transceiver-reciever-pair and last element is the
        reference wmmse allocated power aproximation to compare with computed.
    """
    N = tf.shape(y_pred)[0]
    weights = y_pred[:, -2]
    power = tf.expand_dims(y_pred[:, -3], axis=0)
    sum_rate = compute_sum_rate(power, y_true, weights, N)
    return tf.negative(sum_rate)


@tf.function()
def sum_rate_metric(y_true, y_pred):
    """WMMSE ratio metric.

    Metric function for Radio Resource Management example, computing the sum rate normalized by the
    wmmse sum rate. Inputs are batched tensors with shape (b, n, ?) where b is batch_size, n is the
    number of nodes in the graph an ? changes depending on input.

    Parameters
    ----------
    y_true : tf.Tensor
        Batched tensor with with shape (b, n, n) containing for each node the path losses from the
        node (transceiver-reciever-pair) to all others, include itself.
    y_pred : tf.Tensor
        Batched tensor with GNN output, containing a hidden state with shape (b, n, 4), where second
        last element is the allocated power to transceiver-reciever-pair and last element is the
        reference wmmse allocated power aproximation to compare with computed.
    """
    N = tf.shape(y_pred)[0]
    power_wmmse = tf.expand_dims(y_pred[:, -1], axis=0)
    weights = y_pred[:, -2]
    power = tf.expand_dims(y_pred[:, -3], axis=0)
    sum_rate_wmmse = compute_sum_rate(power_wmmse, y_true, weights, N)
    sum_rate = compute_sum_rate(power, y_true, weights, N)
    return tf.multiply(tf.divide(sum_rate, sum_rate_wmmse), tf.constant(100, dtype=tf.float32))


def main():
    model = ignnition.create_model(model_dir=Path(__file__).parent.absolute())
    model.computational_graph()
    model.train_and_validate()
    # model.predict()


if __name__ == "__main__":
    main()
