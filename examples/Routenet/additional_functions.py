import tensorflow as tf


def normalization(feature, feature_name):
    if feature_name == 'traffic':
        feature = (feature - 373.762) / 229.503
    elif feature_name == 'capacity':
        feature = (feature - 22576.877) / 14802.988
    elif feature_name == 'delay':
        feature = tf.math.log(feature)
    return feature


def denormalization(feature, feature_name):
    if feature_name == 'delay':
        feature = tf.math.exp(feature)
    return feature


def evaluation_metric(label, prediction):
    # Change to proper shapes and compute re
    label = tf.reshape(label, (1, len(label)))
    return tf.math.reduce_mean((label - prediction) / label)
