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

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import ignnition


def normalization(feature, feature_name):
    if feature_name == 'traffic':
        feature = (feature - 170) / 130
    elif feature_name == 'capacity':
        feature = (feature - 25000) / 40000
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
    return tf.math.reduce_mean((label-prediction)/label)

def main():
    model = ignnition.create_model(model_dir='./')
    model.computational_graph()
    model.train_and_validate()
    # model.predict()


if __name__ == "__main__":
    main()
