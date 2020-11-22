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

import configparser
import tensorflow as tf
import datetime
import warnings
import glob
import tarfile
import json
from tensorflow.keras.losses import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.optimizers.schedules import *
import os
from ignnition.gnn_model import Gnn_model
from ignnition.json_preprocessing import Json_preprocessing
from ignnition.data_generator import Generator
from ignnition.utils import *
from ignnition.custom_callbacks import *
import sys
import yaml


class Ignnition_model:
    def __init__(self, path):
        # read the train_options file
        with open(path, 'r') as stream:
            try:
                self.CONFIG = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print("The training options file was not found in " + path)

        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        warnings.filterwarnings("ignore")

        # add the file with any additional function, if any
        if 'additional_functions_file' in self.CONFIG:
            aux = self.CONFIG['additional_functions_file']
            sys.path.insert(1, aux)
            self.module = __import__(aux.split('/')[-1][0:-3])

        self.gnn_model = self.__create_model()
        self.generator = Generator()

        # restore a warm-start Checkpoint (if any)
        self.__restore_model()

    def __loss_function(self, labels, predictions):
        loss_func_name = self.CONFIG['loss']
        try:
            loss_function = getattr(tf.keras.losses, loss_func_name)()
            regularization_loss = sum(self.gnn_model.losses)
            loss = loss_function(labels, predictions)
            total_loss = loss + regularization_loss

        except:  # go to the main file and find the function by this name
            loss_function = getattr(self.module, loss_func_name)
            # loss_function = getattr(self.add_file_name, loss_func_name)
            total_loss = tf.py_function(func=loss_function, inp=[predictions, labels, self.gnn_model], Tout=tf.float32)

        return total_loss

    def __get_keras_metrics(self):  # check
        # do this by default.
        # TODO: Let the user define an array of names of metrics

        # metric_name = ["mae", "mape", "r_squared"]
        metric_name = ["mae", "mape"]
        metrics = []
        for name in metric_name:
            try:
                metrics.append(getattr(tf.keras.metrics, name))

            # if we cannot find this name
            except:
                metrics.append(getattr(self.module, name))

        return metrics

    @tf.autograph.experimental.do_not_convert
    def __get_compiled_model(self, model_info):
        gnn_model = Gnn_model(model_info)

        # dynamically define the optimizer
        #optimizer_params = model_info.get_optimizer()
        optimizer_params = self.CONFIG['optimizer']
        op_type = optimizer_params['type']
        del optimizer_params['type']

        # dynamically define the adaptative learning rate if needed (schedule)
        if 'learning_rate' in optimizer_params and isinstance(optimizer_params['learning_rate'], dict):
            schedule = optimizer_params['learning_rate']
            type = schedule['type']
            del schedule['type']  # so that only the parameters remain
            s = getattr(tf.keras.optimizers.schedules, type)

            # create an instance of the schedule class indicated by the user. Accepts any schedule from keras documentation
            optimizer_params['learning_rate'] = s(**schedule)

        # create the optimizer
        o = getattr(tf.keras.optimizers, op_type)

        # create an instance of the optimizer class indicated by the user. Accepts any loss function from keras documentation
        optimizer = o(**optimizer_params)
        gnn_model.compile(loss=self.__loss_function,
                          optimizer=optimizer,
                          metrics=self.__get_keras_metrics(),
                          run_eagerly=False)
        return gnn_model

    def __get_model_callbacks(self, model_dir, mini_epoch_size, num_epochs, metric_names):
        os.mkdir(model_dir + '/ckpt')

        # HERE WE CAN ADD AN OPTION FOR EARLY STOPPING
        return [tf.keras.callbacks.TensorBoard(log_dir=model_dir + '/logs', update_freq='epoch', write_images=False,
                                               histogram_freq=1),
                tf.keras.callbacks.ModelCheckpoint(filepath=model_dir + '/ckpt/weights.{epoch:02d}-{loss:.2f}.hdf5',
                                                   save_freq='epoch', monitor='loss'),
                Custom_progressbar(model_dir=model_dir + '/logs', mini_epoch_size=mini_epoch_size,
                                   num_epochs=num_epochs, metric_names=metric_names, k=None)]

    # here we pass a mini-batch. We want to be able to perform a normalization over each mini-batch seperately
    def __batch_normalization(self, x, feature_list, output_name, y=None):
        """
        Parameters
        ----------
        x:    tensor
           Tensor with the feature information
        y:    tensor
           Tensor with the label information
        feature_list:    tensor
           List of names with the names of the features in x
        output_names:    tensor
           List of names with the name of the output labels in y
        output_normalizations: dict
           Maps each feature or label with its normalization strategy if any
        """

        # input data
        for f in feature_list:
            f_name = f.name
            # norm_type = f.batch_normalization
            norm_type = 'mean'
            if norm_type == 'mean':
                mean = tf.math.reduce_mean(x.get(f_name))
                variance = tf.math.reduce_std(x.get(f_name))
                x[f_name] = (x.get(f_name) - mean) / variance

            elif norm_type == 'max':
                max = tf.math.reduce_max(x.get(f_name))
                x[f_name] = x.get(f_name) / max
        # output
        if y is not None:
            output_normalization = 'mean'
            if output_normalization == 'mean':
                mean = tf.math.reduce_mean(y)
                variance = tf.math.reduce_std(y)
                y = (y - mean) / variance

            elif output_normalization == 'max':
                max = tf.math.reduce_max(y)
                y = y / max

            return x, y
        return x

    def __global_normalization(self, x, feature_list, output_name, output_normalization, y=None):
        """
        Parameters
        ----------
        x:    tensor
            Tensor with the feature information
        y:    tensor
            Tensor with the label information
        feature_list:    tensor
            List of names with the names of the features in x
        output_names:    tensor
            List of names with the name of the output labels in y
        output_normalizations: dict
            Maps each feature or label with its normalization strategy if any
        """
        # input data
        for f in feature_list:
            f_name = f.name
            norm_type = f.normalization
            if str(norm_type) != str(None):
                try:
                    norm_func = getattr(self.module, norm_type)
                    x[f_name] = tf.py_function(func=norm_func, inp=[x.get(f_name), f_name], Tout=tf.float32)

                except:
                    print_failure('The normalization function ' + str(norm_type) + ' is not defined in the main file.')

        # output
        if y is not None:
            if output_normalization is not None:
                try:
                    norm_func = getattr(self.module, output_normalization)
                    y = tf.py_function(func=norm_func, inp=[y, output_name], Tout=tf.float32)

                except:
                    print_failure('The normalization function ' + str(
                        output_normalization) + ' is not defined in the main file.')
            return x, y

        return x

    @tf.autograph.experimental.do_not_convert
    def __input_fn_generator(self, filenames, shuffle=False, training=True,data_samples=None):
        """
        Parameters
        ----------
        x:    tensor
            Tensor with the feature information
        y:    tensor
            Tensor with the label information
        feature_list:    tensor
            List of names with the names of the features in x
        output_names:    tensor
            List of names with the name of the output labels in y
        output_normalizations: dict
            Maps each feature or label with its normalization strategy if any
        """
        with tf.name_scope('get_data') as _:
            feature_list = self.model_info.get_all_features()
            adjacency_info = self.model_info.get_adjecency_info()
            interleave_list = self.model_info.get_interleave_tensors()
            interleave_sources = self.model_info.get_interleave_sources()
            output_name, output_normalization, _ = self.model_info.get_output_info()
            additional_input = self.model_info.get_additional_input_names()
            unique_additional_input = [a for a in additional_input if a not in feature_list]
            entity_names = self.model_info.get_entity_names()
            types, shapes = {}, {}
            feature_names = []

            for a in unique_additional_input:
                types[a] = tf.int64
                shapes[a] = tf.TensorShape(None)

            for f in feature_list:
                f_name = f.name
                feature_names.append(f_name)
                types[f_name] = tf.float32
                shapes[f_name] = tf.TensorShape(None)

            for a in adjacency_info:
                types['src_' + a[0]] = tf.int64
                shapes['src_' + a[0]] = tf.TensorShape([None])
                types['dst_' + a[0]] = tf.int64
                shapes['dst_' + a[0]] = tf.TensorShape([None])
                types['seq_' + a[1] + '_' + a[2]] = tf.int64
                shapes['seq_' + a[1] + '_' + a[2]] = tf.TensorShape([None])

                if a[3] == 'True':
                    types['params_' + a[0]] = tf.int64
                    shapes['params_' + a[0]] = tf.TensorShape(None)

            for e in entity_names:
                types['num_' + e] = tf.int64
                shapes['num_' + e] = tf.TensorShape([])

            for i in interleave_sources:
                types['indices_' + i[0] + '_to_' + i[1]] = tf.int64
                shapes['indices_' + i[0] + '_to_' + i[1]] = tf.TensorShape([None])

            if training:  # if we do training, we also expect the labels
                if data_samples is None:
                    ds = tf.data.Dataset.from_generator(
                        lambda: self.generator.generate_from_dataset(filenames, entity_names, feature_names,
                                                                     output_name, adjacency_info,
                                                                     interleave_list, unique_additional_input, training,
                                                                     shuffle),
                        output_types=(types, tf.float32),
                        output_shapes=(shapes, tf.TensorShape(None)))
                    ds = ds.repeat()
                else:
                    data_samples = [json.dumps(t) for t in data_samples]
                    ds = tf.data.Dataset.from_generator(
                        lambda: self.generator.generate_from_array(data_samples, entity_names, feature_names,
                                                                   output_name,
                                                                   adjacency_info, interleave_list,
                                                                   unique_additional_input, training, shuffle),
                        output_types=(types, tf.float32),
                        output_shapes=(shapes, tf.TensorShape(None)))

            else:
                if data_samples is None:
                    ds = tf.data.Dataset.from_generator(
                        lambda: self.generator.generate_from_dataset(filenames, entity_names, feature_names,
                                                                     output_name, adjacency_info,
                                                                     interleave_list, unique_additional_input, training,
                                                                     shuffle),
                        output_types=(types),
                        output_shapes=(shapes))

                else:
                    data_samples = [json.dumps(t) for t in data_samples]
                    ds = tf.data.Dataset.from_generator(
                        lambda: self.generator.generate_from_array(data_samples, entity_names, feature_names,
                                                                   output_name,
                                                                   adjacency_info, interleave_list,
                                                                   unique_additional_input, training, shuffle),
                        output_types=(types),
                        output_shapes=(shapes))

            with tf.name_scope('normalization') as _:
                global_norm = True
                if global_norm:
                    if training:
                        ds = ds.map(
                            lambda x, y: self.__global_normalization(x, feature_list, output_name, output_normalization,
                                                                     y),
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
                        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

                    else:
                        ds = ds.map(
                            lambda x: self.__global_normalization(x, feature_list, output_name, output_normalization),
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
                        ds = iter(ds)

                else:
                    if training:
                        ds = ds.map(lambda x, y: self.__batch_normalization(x, feature_list, output_name, y),
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
                        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

                    else:
                        ds = ds.map(lambda x: self.__batch_normalization(x, feature_list, output_name),
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
                        ds = iter(ds)

        return ds

    def __make_model(self, model_info):
        # Either restore the latest model, or create a fresh one
        print("Creating a new model")
        gnn_model = self.__get_compiled_model(model_info)

        return gnn_model

    # FUNCTIONALITIES
    def train_and_evaluate(self, training_samples=None, eval_samples=None):
        # training_files is a list of strings (paths)
        # eval_files is a list of strings (paths)
        print()
        print_header(
            'Starting the training and validation process...\n---------------------------------------------------------------------------\n')

        filenames_train = os.path.normpath(self.CONFIG['train_dataset'])
        filenames_eval = os.path.normpath(self.CONFIG['validation_dataset'])

        model_dir = os.path.normpath(self.CONFIG['model_dir'])

        model_dir = os.path.join(model_dir, 'CheckPoint')

        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)

        model_dir = os.path.join(model_dir, 'experiment_' + str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")))
        os.mkdir(model_dir)

        strategy = tf.distribute.MirroredStrategy()  # change this not to use GPU
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        train_dataset = self.__input_fn_generator(filenames_train,
                                                  shuffle=str_to_bool(
                                                      self.CONFIG['shuffle_training_set']),
                                                  data_samples=training_samples)
        validation_dataset = self.__input_fn_generator(filenames_eval,
                                                       shuffle=str_to_bool(
                                                           self.CONFIG['shuffle_validation_set']),
                                                       data_samples=eval_samples)

        mini_epoch_size = None if self.CONFIG['epoch_size'] == 'All' else int(
            self.CONFIG['epoch_size'])
        num_epochs = int(self.CONFIG['epochs'])
        metrics = self.CONFIG['metrics']
        # pass the validation data to the callback and do this manually??
        callbacks = self.__get_model_callbacks(model_dir=model_dir, mini_epoch_size=mini_epoch_size,
                                               num_epochs=num_epochs, metric_names=metrics)

        self.gnn_model.fit(train_dataset,
                           epochs=num_epochs,
                           steps_per_epoch=mini_epoch_size,
                           batch_size= self.CONFIG.get('batch_size', 1),
                           validation_data=validation_dataset,
                           validation_freq=int(self.CONFIG['val_frequency']),
                           validation_steps=int(self.CONFIG['val_samples']),
                           callbacks=callbacks,
                           use_multiprocessing=True,
                           verbose=0)

    def __create_model(self):
        model_description_path = self.CONFIG['model_description_path']
        dimensions, len1_features = self.find_dataset_dimensions(self.CONFIG['train_dataset'])
        self.model_info = Json_preprocessing(model_description_path, dimensions, len1_features)  # read json

        return self.__make_model(self.model_info)

    def __restore_model(self):
        if 'warm_start_path' in self.CONFIG:
            checkpoint_path = self.CONFIG['warm_start_path']
        else:
            checkpoint_path = ''

        if os.path.isfile(checkpoint_path):
            print("Restoring from", checkpoint_path)
            # in this case we need to initialize the weights to be able to use a warm-start checkpoint
            sample_it = self.__input_fn_generator(self.CONFIG['train_dataset'], training=False,
                                                  data_samples=None)
            sample = sample_it.get_next()

            # Call only one tf.function when tracing.
            _ = self.gnn_model(sample, training=False)

            return self.gnn_model.load_weights(checkpoint_path)



    def find_dataset_dimensions(self, path):
        """
        Parameters
        ----------
        path:    str
          Path to find the dataset
        """
        sample = glob.glob(str(path) + '/*.tar.gz')[0]  # choose one single file to extract the dimensions
        try:
            tar = tarfile.open(sample, 'r:gz')  # read the tar files
        except:
            print_failure('The file data.json was not found in ' + sample)

        try:
            file_samples = tar.extractfile('data.json')
            file_samples.read(1)
            aux = stream_read_json(file_samples)

            sample_data = next(aux)  # read one single example #json.load(file_samples)[0]  # one single sample
            dimensions = {}  # for each key, we have a tuple of (length, num_elements)

            # note that all the features that are 1d will have dimension 1. o/w it has 2 dimensions
            len_1_features = []
            for k, v in sample_data.items():
                # if it's a feature
                if not isinstance(v, dict):
                    if isinstance(v, list):
                        if isinstance(v[0], str):
                            pass

                        # if it's a feature (2-d array)
                        elif isinstance(v[0], list):
                            dimensions[k] = len(v[0])

                        # set always to len(v). In run time, if its a 1-d feature of a node, replace for dimension = 1
                        else:
                            dimensions[k] = len(v)
                            len_1_features.append(k)

                    # if it is one single value
                    else:
                        dimensions[k] = 1

                # if its either the entity or an adjacency (it is a dictionary, that is non-empty)
                elif v:
                    first_key = list(v.keys())[0]  # first key of the list
                    element = v[first_key]  # first value of the list (another list)
                    if (not isinstance(element[0], str)) and isinstance(element[0], list):
                        # the element[0][0] is the adjacency node. The element[0][1] is the edge information
                        dimensions[k] = len(element[0][1])
                    else:
                        dimensions[k] = 0

            return dimensions, len_1_features

        except:
            print_failure('Failed to read the data file ' + sample)

    def predict(self, prediction_samples=None):
        """
            Parameters
            ----------
            model_info:    object
            Object with the json information model
        """
        print()
        print_header('Starting to make the predictions...\n---------------------------------------------------------\n')

        if prediction_samples is None:
            try:
                data_path = self.CONFIG['predict_dataset']
            except:
                print_failure(
                    'Make sure to either pass an array of samples or to define in the train_options.ini the path to the prediction dataset')

        else:
            data_path = None

        sample_it = self.__input_fn_generator(data_path, training=False, data_samples=prediction_samples)
        all_predictions = []
        try:
            # while there are predictions
            while True:
                pred = self.gnn_model(sample_it.get_next(), training=False)
                pred = tf.squeeze(pred)
                output_name, _, output_denormalization = self.model_info.get_output_info()  # for now suppose we only have one output type
                try:
                    denorm_func = getattr(self.module, output_denormalization)
                    pred = tf.py_function(func=denorm_func, inp=[pred, output_name], Tout=tf.float32)
                except:
                    print_info(
                        'A denormalization function for output ' + output_name + ' was not defined. The output will be normalized.')

                all_predictions.append(pred)

        except tf.errors.OutOfRangeError:
            pass

        return all_predictions

    def computational_graph(self):
        print()
        print_header(
            'Generating the computational graph... \n---------------------------------------------------------\n')

        filenames_train = os.path.normpath(self.CONFIG['train_dataset'])

        path = os.path.normpath(self.CONFIG['model_dir'])

        path = os.path.join(path, 'computational_graphs',
                            'experiment_' + str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")))

        # tf.summary.trace_on() and tf.summary.trace_export().
        writer = tf.summary.create_file_writer(path)
        tf.summary.trace_on(graph=True, profiler=True)

        # evaluate one single input
        sample_it = self.__input_fn_generator(filenames_train, training=False, data_samples=None)
        sample = sample_it.get_next()
        # Call only one tf.function when tracing.
        _ = self.gnn_model(sample, training=False)

        with writer.as_default():
            tf.summary.trace_export(
                name="computational_graph_" + str(datetime.datetime.now()),
                step=0,
                profiler_outdir=path)
