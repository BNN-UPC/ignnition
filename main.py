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

#def normalization(features):
#   features['traffic'] = (features['traffic'] - 170) / 130
#   features['link_capacity'] = (features['link_capacity'] - 25000) / 40000
#   features['delay'] = tf.math.log(features['delay'])
#   return features

def normalization(feature, feature_name):
   if feature_name == 'traffic':
        feature = (feature - 170) / 130
   elif feature_name == 'link_capacity':
        feature = (feature - 25000) / 40000
   elif feature_name == 'delay':
        feature = tf.math.log(feature)
   return feature

def denormalization(feature, feature_name):
    if feature_name == 'delay':
        feature = tf.math.exp(feature)
    return feature

import tarfile
import json
import glob
import time
def main():
    model = ignnition.create_model(model_dir = './examples/Routenet')
    #model.computational_graph()
    model.train_and_validate()
    #model.predict()

    #evaluation_files = glob.glob(
    #     '../Datasets/Datasets_framework/Dataset_routenet/eval/*.tar.gz')

    #results = {}
    #for f in evaluation_files:
    #    tar = tarfile.open(f, 'r:gz')  # read the tar files
    #    file_samples = tar.extractfile('data.json')
    #    samples = json.load(file_samples)
    #    model.batch_training(samples[0:32])
    #    evaluation_metrics = model.evaluate(samples)

         # for each sample in the file
         #counter = 0
         #for e in evaluation_metrics:
         #    mre = np.mean(e)
         #    results[f + '_sample_' + str(counter)] = mre

    #
    # print(results)
    # sorted_samples = {k:v for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)}
    # print(sorted_samples)
    # k_top = sorted_samples.keys()[0:10]
    # print("The k top are: ", k_top)


if __name__ == "__main__":
        main ()
