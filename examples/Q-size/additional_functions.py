import tensorflow as tf


def normalization(feature, feature_name):
    if feature_name == 'delay':
        feature = (tf.math.log(feature) + 1.78) / 0.93
    if feature_name == 'traffic':
        feature = (feature - 0.28) / 0.15
    if feature_name == 'jitter':
        feature = (feature - 1.5) / 1.5
    if feature_name == 'link_capacity':
        feature = (feature - 27.0) / 14.86
    if feature_name == 'queue_sizes':
        feature = (feature - 16.5) / 15.5
    return feature
