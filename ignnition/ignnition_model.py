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
from ignnition.yaml_preprocessing import Yaml_preprocessing
from ignnition.data_generator import Generator
from ignnition.utils import *
from ignnition.custom_callbacks import *
import sys
import yaml
import collections
import networkx as nx
from networkx.readwrite import json_graph
from itertools import chain


class Ignnition_model:
    """
    This class implements the main interface to execute the framework. It includes the main functionalities of the training, which can be called by the user. Additionally, it incorporates all the necessary functionalities to create/restore a model.

    Attributes
    ----------
    model_dir:    str
        Path to the model directory where the train_options is found as well as all the other files.
    CONFIG: dict
        Dictionary containing all the information from the train_options.yaml file.
    module: obj
        Import of the module to be used to call custom functions such as normalization ones.
    model_info: yaml_preprocessing
        Object which is in charge of handling all the information of the model_description file.
    generator: Generator obj
        Object in charge of feeding the data to the model.

    Methods:
    ----------
    __process_path(self, path)
        This method takes as input a path and, considering the location of the model directory, converts all the relative path to absolute paths starting from such model_directory

    __loss_function(self, labels, predictions)
       Function that calls executes the loss function object from the keras libarary if specified. O/w it looks for a custom loss function specified in the module file.

    __get_keras_metrics(self)
        Creates all the keras metrics corresponding to tf.keras objects, or it creates objects based on custom function specified in the module path.

    __get_compiled_model(self, model_info)
        Compiles the tf model with all the corresponding options

    __get_model_callbacks(self, output_path, mini_epoch_size, num_epochs, metric_names)
        Creates all the callbacks (these being the k-best (if specified), model checkpoints, and tensorboard)

    __batch_normalization(self, x, feature_list, y=None)
        Performs batch normalization on the data (e.g., normalizes all the batch by its max, min..)

    __global_normalization(self, x, feature_list, output_name, y=None)
        Performs a global normalization operation which must be specified in the module path (all the samples are normalized according to the same criteria).

    __input_fn_generator(self, filenames=None, shuffle=False, training=True,data_samples=None, iterator=False)
        Method that creates the dataset which is served by the generator that we created before.

    __create_model(self)
        Method that creates the yaml_preprocessing object that processed the model_description file and creates the subsequent classes to organize the info.

    __create_gnn(self,samples=None, path=None, verbose=True)
        Creates the GNN object itself.

    __restore_model(self, gnn_model, sample)
        Restores the weights from a GNN that is saved in the given path to the current GNN model.

    find_dataset_dimensions(self, path=None, samples=None)
        Looks for the first training samples and processes it to extract the dimensions of all the input tensors (necessary to create the GNN model)s

    train_and_validate(self, training_samples=None, eval_samples=None)
        Public operation that is called by the user to initiate a training and validation operation of the current GNN model.

    predict(self, prediction_samples=None, verbose=True)
        Public operation that is callable by the user to initiate a predict operatio of a given array of data/dataset using the current GNN model.

    computational_graph(self)
        Public method callable by the user to create a computation graph of the desired model which can be then used for debugging purposes.

    evaluate(self, evaluation_samples = None, verbose=True)
        Public method callable by the user that executes an evaluation functionality given some metrics.

    batch_training(self, input_samples)
        Public method callable by the user, useful in RL context, to execute a training of a single batch of data. No verbosite is set.
    """

    def __init__(self, model_dir):
        """
        Parameters
        ----------
        model_dir:    str
           Path to the model directory
        """
        path = os.path.normpath(model_dir)
        self.model_dir = os.path.abspath(path)

        train_options_path = os.path.join(self.model_dir, 'train_options.yaml')

        # read the train_options file
        with open(train_options_path, 'r') as stream:
            try:
                self.CONFIG = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print("The training options file was not found in " + train_options_path)

        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        warnings.filterwarnings("ignore")

        # add the file with any additional function, if any
        if 'additional_functions_file' in self.CONFIG:
            additional_path = self.__process_path(self.CONFIG['additional_functions_file'])
            sys.path.insert(1, additional_path)
            self.module = __import__(additional_path.split('/')[-1][0:-3])

        self.model_info = self.__create_model()
        self.generator = Generator()

    def __process_path(self, path):
        """
        Parameters
        ----------
        path:    string
           Normalized or absolute path to be converted
        """
        return os.path.normpath(os.path.join(self.model_dir, path))

    def __loss_function(self, labels, predictions):
        """
        Parameters
        ----------
        labels:    tensor
           Input label
        predictions:    tensor
           Predictions of the GNN model
        """

        loss_func_name = self.CONFIG['loss']
        try:
            loss_function = getattr(tf.keras.losses, loss_func_name)()
            regularization_loss = sum(self.gnn_model.losses)
            loss = loss_function(labels, predictions)
            total_loss = loss + regularization_loss

        except:  # go to the main file and find the function by this name
            loss_function = getattr(self.module, loss_func_name)
            total_loss = tf.py_function(func=loss_function, inp=[predictions, labels, self.gnn_model], Tout=tf.float32)

        return total_loss

    def __get_keras_metrics(self):
        metric_names = self.CONFIG['metrics']
        metrics = []
        for name in metric_names:
            if hasattr(tf.keras.metrics, name):
                metrics.append(getattr(tf.keras.metrics, name)())
            elif hasattr(self.module, name):
                metrics.append(getattr(self.module, name))

        return metrics

    @tf.autograph.experimental.do_not_convert
    def __get_compiled_model(self, model_info):
        """
        Parameters
        ----------
        model_info:    Yaml_preprocessing object
            Object in charge of handling the information in the model_description.yaml file
        """

        gnn_model = Gnn_model(model_info)

        # dynamically define the optimizer
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

    def __get_model_callbacks(self, output_path):
        """
        Parameters
        ----------
        output_path:    str
            Path where the checkpoint files and logs are saved
        """

        os.mkdir(output_path + '/ckpt')

        # HERE WE CAN ADD AN OPTION FOR EARLY STOPPING
        return [tf.keras.callbacks.TensorBoard(log_dir=output_path + '/logs', update_freq='epoch', write_images=False,
                                               histogram_freq=1),
                tf.keras.callbacks.ModelCheckpoint(filepath=output_path + '/ckpt/weights.{epoch:02d}-{loss:.2f}.hdf5',
                                                   save_freq='epoch', monitor='loss'),
                K_best(output_path=output_path + '/logs', k=self.CONFIG.get('k_best', None))]

    # here we pass a mini-batch. We want to be able to perform a normalization over each mini-batch seperately
    def __batch_normalization(self, x, feature_list, norm_type, y=None):
        """
        Parameters
        ----------
        x:    tensor
           Tensor with the feature information

        feature_list:    tensor
           List of names with the names of the features in x
        norm_type: string
            Defines the type of batch normalization to be used
        y:    tensor
           Tensor with the label information
        """

        # input data
        for f in feature_list:
            f_name = f.name
            # norm_type = f.batch_normalization
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

    def __global_normalization(self, x, feature_list, output_name, y=None):
        """
        Parameters
        ----------
        x:    tensor
            Tensor with the feature information
        feature_list:    tensor
            List of names with the names of the features in x
        output_names:    tensor
            List of names with the name of the output labels in y
        y:    tensor
            Tensor with the label information
        """

        try:
            norm_func = getattr(self.module, 'normalization')
        except:
            norm_func = None

        if norm_func is not None:
            # input data
            for f_name in feature_list:
                try:
                    x[f_name] = tf.py_function(func=norm_func, inp=[x.get(f_name), f_name], Tout=tf.float32)
                except:
                    print_failure('The normalization function failed with feature ' + f_name + '.')

            # output
            if y is not None:
                try:
                    y = tf.py_function(func=norm_func, inp=[y, output_name], Tout=tf.float32)
                except:
                    print_failure('The normalization function computing the output label' + output_name + ' failed.')
                return x, y
            return x

        if y is not None:
            return x, y
        return x

    @tf.autograph.experimental.do_not_convert
    def __input_fn_generator(self, filenames=None, shuffle=False, training=True, data_samples=None, iterator=False):
        """
        Parameters
        ----------
        filenames:    string
            Tensor with the filenames of the input (if using dataset input only)
        shuffle:    bool
            Bool indicating if we need to shuffle the input data.
        training:    bool
            Bool indicating if we are performing a training operation (and thus a label is expected)
        data_samples:    [array]
            List of samples to be used as input (if any)
        iterator: bool
            Indicates if we need to transform the dataset to an iterator
        """

        with tf.name_scope('get_data') as _:
            feature_list = self.model_info.get_all_features()
            adj_names = self.model_info.get_adjacency_info()
            interleave_list = self.model_info.get_interleave_tensors()
            interleave_sources = self.model_info.get_interleave_sources()
            output_name = self.model_info.get_output_info()
            additional_input = self.model_info.get_additional_input_names()
            unique_additional_input = [a for a in additional_input if a not in feature_list]
            entity_names = self.model_info.get_entity_names()
            types, shapes = {}, {}
            feature_names = []

            for a in unique_additional_input:
                types[a] = tf.int64
                shapes[a] = tf.TensorShape(None)

            for f_name in feature_list:
                feature_names.append(f_name)
                types[f_name] = tf.float32
                shapes[f_name] = tf.TensorShape(None)

            for a in adj_names:
                types['src_' + a] = tf.int64
                shapes['src_' + a] = tf.TensorShape([None])
                types['dst_' + a] = tf.int64
                shapes['dst_' + a] = tf.TensorShape([None])
                types['seq_' + a] = tf.int64
                shapes['seq_' + a] = tf.TensorShape([None])

                # we now include this values in the additional_params
            # if a[3] == 'True':
            #     types['params_' + a[0]] = tf.int64
            #     shapes['params_' + a[0]] = tf.TensorShape(None)

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
                                                                     output_name,  # adjacency_info,
                                                                     interleave_list, unique_additional_input, training,
                                                                     shuffle),
                        output_types=(types, tf.float32),
                        output_shapes=(shapes, tf.TensorShape(None)))
                    ds = ds.repeat()
                else:
                    data_samples = [json.dumps(t) for t in data_samples]
                    ds = tf.data.Dataset.from_generator(
                        lambda: self.generator.generate_from_array(data_samples, entity_names, feature_names,
                                                                   output_name,  # adjacency_info,
                                                                   interleave_list,
                                                                   unique_additional_input, training, shuffle),
                        output_types=(types, tf.float32),
                        output_shapes=(shapes, tf.TensorShape(None)))

            else:
                if data_samples is None:
                    ds = tf.data.Dataset.from_generator(
                        lambda: self.generator.generate_from_dataset(filenames, entity_names, feature_names,
                                                                     output_name,  # adjacency_info,
                                                                     interleave_list, unique_additional_input, training,
                                                                     shuffle),
                        output_types=(types),
                        output_shapes=(shapes))

                else:
                    data_samples = [json.dumps(t) for t in data_samples]
                    ds = tf.data.Dataset.from_generator(
                        lambda: self.generator.generate_from_array(data_samples, entity_names, feature_names,
                                                                   output_name,  # adjacency_info,
                                                                   interleave_list,
                                                                   unique_additional_input, training, shuffle),
                        output_types=(types),
                        output_shapes=(shapes))

            with tf.name_scope('normalization') as _:
                batch_norm = self.CONFIG.get('batch_normalization', None)
                if batch_norm is None:
                    if training:
                        ds = ds.map(
                            lambda x, y: self.__global_normalization(x, feature_list, output_name, y),
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
                        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

                    else:
                        ds = ds.map(
                            lambda x: self.__global_normalization(x, feature_list, output_name),
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
                        ds = iter(ds)

                else:
                    if training:
                        ds = ds.map(lambda x, y: self.__batch_normalization(x, feature_list, batch_norm, y),
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
                        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

                    else:
                        ds = ds.map(lambda x: self.__batch_normalization(x, feature_list, batch_norm),
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)

            if iterator:
                ds = iter(ds)

        return ds

    # -------------------------------------
    def __create_model(self):
        print_header(
            "\nProcessing the described model...\n---------------------------------------------------------------------------\n")
        return Yaml_preprocessing(self.model_dir)  # read json

    def __create_gnn(self, samples=None, path=None, verbose=True):
        """
        Parameters
        ----------
        samples:    [array]
            Array of samples to be used as input (if any)
        path:    bool
            Path to find the input data (applicable only if using dataset input)
        verbose:    bool
            Indicates if we want verbosity in the prints of the terminal
        """

        if verbose:
            print_header(
                "Creating the GNN model...\n---------------------------------------------------------------------------\n")

        dimensions, sample = self.find_dataset_dimensions(samples=samples, path=path)
        self.model_info.add_dimensions(dimensions)

        gnn_model = self.__get_compiled_model(self.model_info)
        # restore a warm-start Checkpoint (if any)
        self.gnn_model = self.__restore_model(gnn_model, sample=sample)

    def __restore_model(self, gnn_model, sample):
        """
        Parameters
        ----------
        gnn_model:    GNN obj
            GNN obj of the actual model
        sample:    dict
            Input dictionary necessary to initialize all the dimensions
        """

        checkpoint_path = self.CONFIG.get('warm_start_path', '')
        if os.path.isfile(checkpoint_path):
            print("Restoring from", checkpoint_path)
            # in this case we need to initialize the weights to be able to use a warm-start checkpoint

            sample_it = self.__input_fn_generator(training=False,
                                                  data_samples=[sample])
            sample = sample_it.get_next()
            # Call only one tf.function when tracing.
            _ = gnn_model(sample, training=False)
            gnn_model.load_weights(checkpoint_path)

        elif checkpoint_path != '':
            print_info(
                "The file in the directory " + checkpoint_path + ' was not a valid checkpoint file in hdf5 format.')

        return gnn_model

    # --------------------------------
    def find_dataset_dimensions(self, path=None, samples=None):
        """
        Parameters
        ----------
        path:    str
          Path to find the dataset
        samples: [array]
            Array of samples to be used as input (if any)
        """

        if samples is not None:
            sample = samples[0]  # take the first one to find the dimensions

        else:
            sample_paths = (glob.glob(path + '/*.tar.gz') + glob.glob(path + '/*.json'))

            if sample_paths == []:
                print_failure("No dataset found. Please make sure the paths of the datasets are correct.")
            else:
                sample_path = sample_paths[0] # choose one single file to extract the dimensions

            if '.tar.gz' in sample_path:
                try:
                    tar = tarfile.open(sample_path, 'r:gz')  # read the tar files
                    member = tar.getmembers()[0]
                    file_samples = tar.extractfile(member)
                except:
                    print_failure('The tar file ' + sample_path + ' could not be opened')

            # if it is already a json file
            else:
                file_samples = open(sample_path, 'r')

            try:
                file_samples.read(1)
                aux = stream_read_json(file_samples)
                sample = next(aux)  # read one single example #json.load(file_samples)[0]  # one single sample

            except:
                print_failure('Failed to read the data file ' + sample)

            # Now that we have the sample, we can process the dimensions
            dimensions = {}  # for each key, we have a tuple of (length, num_elements)

            # COMPUTE THE DIMENSIONS USING ONE OF THE SAMPLES
            # 1) Transform it to networkx
            # 2) Obtain all the nodes attributes
            # 3) Obtain all the edge attributes
            # 4) Obtain all the graph attributes

            # 1) Obtain the corresponding graph
            G = json_graph.node_link_graph(sample)

            # 1) Node attributes
            node_attrs = list(set(chain.from_iterable(d.keys() for _, d in G.nodes(data=True))))
            for n in node_attrs:
                if n != 'entity:':
                    features = list(nx.get_node_attributes(G, n).values())
                    elem = features[0]
                    # if features has dimension 1, then dim = 1.
                    if isinstance(elem, list):
                        dimensions[n] = len(elem)
                    else:
                        dimensions[n] = 1

            # 2) Edge attributes
            edge_attrs = list(set(chain.from_iterable(d.keys() for *_, d in G.edges(data=True))))
            for e in edge_attrs:
                features = list(nx.get_edge_attributes(G, e).values())
                if isinstance(features[0], list):
                    dimensions[e] = len(features[0])
                else:
                    dimensions[e] = 1

            # 3) Graph attributes
            graph_attrs = list(G.graph.keys())
            for g in graph_attrs:
                feature = G.graph[g]
                dimensions[g] = len(feature)

            return dimensions, sample

    # FUNCTIONALITIES
    # --------------------------------------------------
    def train_and_validate(self, training_samples=None, val_samples=None):
        """
        Parameters
        ----------
        training_samples:    [array]
            Array of input training samples, if no dataset is used.
        val_samples:    [array]
            Array of input validation samples, if no dataset is used.
        """

        # Create the GNN model
        if not hasattr(self, 'gnn_model'):
            if training_samples is None:  # look for the dataset path
                training_path = self.__process_path(self.CONFIG['train_dataset'])
                self.__create_gnn(path=training_path)
            else:
                self.__create_gnn(samples=training_samples)

        print()
        print_header(
            'Starting the training and validation process...\n---------------------------------------------------------------------------\n')

        filenames_train = self.__process_path(self.CONFIG['train_dataset'])
        filenames_val = self.__process_path(self.CONFIG['validation_dataset'])

        output_path = self.__process_path(self.CONFIG['output_path'])
        output_path = os.path.join(output_path, 'CheckPoint')

        if not os.path.isdir(output_path):
            os.mkdir(output_path)

        output_path = os.path.join(output_path,
                                   'experiment_' + str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")))
        os.mkdir(output_path)

        strategy = tf.distribute.MirroredStrategy()  # change this not to use GPU
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        train_dataset = self.__input_fn_generator(filenames_train,
                                                  shuffle=str_to_bool(
                                                      self.CONFIG['shuffle_training_set']),
                                                  data_samples=training_samples)
        validation_dataset = self.__input_fn_generator(filenames_val,
                                                       shuffle=str_to_bool(
                                                           self.CONFIG['shuffle_validation_set']),
                                                       data_samples=val_samples)

        mini_epoch_size = self.CONFIG.get('epoch_size', None)
        if mini_epoch_size is not None:
            mini_epoch_size = int(mini_epoch_size)

        num_epochs = int(self.CONFIG['epochs'])

        callbacks = self.__get_model_callbacks(output_path=output_path)

        self.gnn_model.fit(train_dataset,
                           epochs=num_epochs,
                           initial_epoch=self.CONFIG.get('initial_epoch', 0),
                           steps_per_epoch=mini_epoch_size,
                           batch_size=self.CONFIG.get('batch_size', 1),
                           validation_data=validation_dataset,
                           validation_freq=int(self.CONFIG['val_frequency']),
                           validation_steps=int(self.CONFIG['val_samples']),
                           callbacks=callbacks,
                           use_multiprocessing=True,
                           verbose=1)

    def predict(self, prediction_samples=None, verbose=True):
        """
        Parameters
        ----------
        prediction_samples:    [array]
            Array of samples to be used for prediction, useful only if no prediction dataset is specified.
        verbose: bool
            Indicates if there should be verbosity in the prints of the terminal or not.
        """

        prediction_path = None
        if not hasattr(self, 'gnn_model'):
            if prediction_samples is None:  # look for the dataset path
                try:
                    prediction_path = self.__process_path(self.CONFIG['predict_dataset'])
                    self.__create_gnn(path=prediction_path, verbose=verbose)
                except:
                    print_failure(
                        'Make sure to either pass an array of samples or to define in the train_options.yaml the path to the prediction dataset')

            else:
                self.__create_gnn(samples=prediction_samples, verbose=verbose)

        if verbose:
            print()
            print_header(
                'Starting to make the predictions...\n---------------------------------------------------------\n')

        sample_it = self.__input_fn_generator(prediction_path, training=False, data_samples=prediction_samples,
                                              iterator=True)
        all_predictions = []
        try:
            # find the denormalization function
            try:
                denorm_func = getattr(self.module, 'denormalization')
            except:
                denorm_func = None

            # while there are predictions
            while True:
                pred = self.gnn_model(sample_it.get_next(), training=False)
                pred = tf.squeeze(pred)
                output_name = self.model_info.get_output_info()  # for now suppose we only have one output type

                if denorm_func is not None:
                    try:
                        pred = tf.py_function(func=denorm_func, inp=[pred, output_name], Tout=tf.float32)
                    except:
                        print_failure('The denormalization function failed')

                all_predictions.append(pred)

        except tf.errors.OutOfRangeError:
            pass

        return all_predictions

    def computational_graph(self):
        # Check if we can generate the computational graph without a dataset
        train_path = self.__process_path(self.CONFIG['train_dataset'])
        if not hasattr(self, 'gnn_model'):
            self.__create_gnn(path=train_path)

        print()
        print_header(
            'Generating the computational graph... \n---------------------------------------------------------------------------\n')

        path = self.__process_path(self.CONFIG['output_path'])

        path = os.path.join(path, 'computational_graphs',
                            'experiment_' + str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")))

        # tf.summary.trace_on() and tf.summary.trace_export().
        writer = tf.summary.create_file_writer(path)
        tf.summary.trace_on(graph=True, profiler=True)

        # evaluate one single input
        sample_it = self.__input_fn_generator(train_path, training=False, data_samples=None, iterator=True)
        sample = sample_it.get_next()
        # Call only one tf.function when tracing.
        _ = self.gnn_model(sample, training=False)

        with writer.as_default():
            tf.summary.trace_export(
                name="computational_graph_" + str(datetime.datetime.now()),
                step=0,
                profiler_outdir=path)

    def evaluate(self, evaluation_samples=None, verbose=True):
        """
        Parameters
        ----------
        evaluation_samples:    [array]
            Array of samples to be used for evaluation, useful only if no prediction dataset is specified.
        verbose: bool
            Indicates if there should be verbosity in the prints of the terminal or not.
        """

        # Generate the model if it doesn't exist
        if not hasattr(self, 'gnn_model'):
            if evaluation_samples is None:  # look for the dataset path
                val_path = self.__process_path(self.CONFIG['validation_dataset'])
                self.__create_gnn(path=val_path)
            else:
                self.__create_gnn(samples=evaluation_samples)

        if verbose:
            print()
            print_header('Starting to make evaluations...\n---------------------------------------------------------\n')

        if evaluation_samples is None:
            try:
                data_path = self.CONFIG['validation_dataset']
            except:
                print_failure(
                    'Make sure to either pass an array of samples or to define in the train_options.yaml the path to the validation dataset')
        else:
            data_path = None

        sample_it = self.__input_fn_generator(data_path, training=True, data_samples=evaluation_samples, iterator=True)

        all_metrics = []
        try:
            try:
                denorm_func = getattr(self.module, 'denormalization')
            except:
                denorm_func = None

            # metric for the evaluation
            try:
                metric_func = getattr(self.module, 'evaluation_metric')

            except:
                print_failure('The evaluation metric function failed. '
                              'Please make sure you define a valid python function taking as input the label '
                              'and the prediction, and returning one single numerical value.')

            # while there are predictions
            while True:
                features, label = sample_it.get_next()
                pred = self.gnn_model(features, training=False)
                pred = tf.squeeze(pred)
                output_name = self.model_info.get_output_info()  # for now suppose we only have one output type
                if denorm_func is not None:
                    try:
                        pred = tf.py_function(func=denorm_func, inp=[pred, output_name], Tout=tf.float32)
                        label = tf.py_function(func=denorm_func, inp=[label, output_name], Tout=tf.float32)
                    except:
                        print_failure('The denormalization function failed')

                # compute the metric value
                value = tf.py_function(func=metric_func, inp=[label, pred], Tout=tf.float32)
                all_metrics.append(value)

        except tf.errors.OutOfRangeError:
            pass
        return all_metrics

    def batch_training(self, input_samples):
        """
        Parameters
        ----------
        input_samples:    [array]
           Array of samples to be used for training (following the same format as if they were in a dataset)
        """

        if not hasattr(self, 'gnn_model'):
            self.__create_gnn(samples=input_samples)

        dataset = self.__input_fn_generator(None, training=True, data_samples=input_samples, iterator=False)
        self.gnn_model.fit(dataset, batch_size=len(input_samples), verbose=0)
