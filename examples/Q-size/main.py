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
sys.path.append('./utils/')
import framework_operations as ignnition


def normalization_queue_size(feature, feature_name):
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

def main():
    model = ignnition.create_model()
    ignnition.debug(model)
    ignnition.train_and_evaluate(model)
    #ignnition.predict(model)


if __name__ == "__main__":
        main ()