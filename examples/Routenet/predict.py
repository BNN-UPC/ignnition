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
import numpy as np
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import ignnition

def main():
    model = ignnition.create_model(model_dir='./')
    model.computational_graph()
    all_metrics = model.evaluate()

    convert_to_np = []
    for elem in all_metrics:
        convert_to_np.append(elem.numpy())

    with open('Results.pkl', 'wb') as f:
        pickle.dump(convert_to_np, f)


if __name__ == "__main__":
    main()
