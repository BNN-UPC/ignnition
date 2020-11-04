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

import sys
import tensorflow as tf
import ignnition


def normalization_features(feature, feature_name):
    if feature_name == 'link_utilizations':
        feature = (feature) / 228.0
    elif feature_name == 'traffic':
        feature = (feature - 9.949) / 0.845
    return feature


def normalization_func(feature, feature_name):
    return (feature - 0.3984850402191232) / 0.2104260838365741

def denormalization_func(feature, feature_name):
    return feature * 0.2104260838365741 + 0.3984850402191232

def normalization_routenet(feature, feature_name):
    if feature_name == 'traffic':
        feature = (feature - 170) / 130
    if feature_name == 'link_capacity':
        feature = (feature - 25000) / 40000

    return feature

def log(feature, feature_name):
    return tf.math.log(feature)

def exp(feature, feature_name):
    return tf.math.exp(feature)


def r_squared(y_true, y_pred):
    total_error = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    unexplained_error = tf.reduce_sum(tf.square(y_true - y_pred))
    r_sq = 1.0 - tf.truediv(unexplained_error, total_error)
    return tf.reduce_mean(r_sq)


import json
def main():
    model = ignnition.create_model('./train_options.ini')
    model.computational_graph()
    model.train_and_evaluate()


if __name__ == "__main__":
        main ()
