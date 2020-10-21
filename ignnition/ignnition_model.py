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
import inspect
import numpy as np
from ignnition.gnn_model import Gnn_model
from ignnition.json_preprocessing import Json_preprocessing
from ignnition.data_generator import Generator
import main

class Ignnition_model:
    def __init__(self, path):
        self.end_symbol = bytes(']', 'utf-8')
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        self.CONFIG = configparser.ConfigParser()
        self.CONFIG._interpolation = configparser.ExtendedInterpolation()
        self.CONFIG.read(path)

        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
        warnings.filterwarnings("ignore")

        self.model = self.create_model()
        self.generator = Generator()


    def normalization(self, x, feature_list, output_name, output_normalization, y=None):

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
            if str(norm_type) != 'None':
                try:
                    x[f_name] = getattr(main, norm_type)(x[f_name], f_name)
                except:
                    tf.compat.v1.logging.error(
                        'IGNNITION: The normalization function ' + str(norm_type) + ' is not defined in the main file.')
                    sys.exit(1)

        # output
        if y != None:
            if str(output_normalization) != 'None':
                try:
                    y = getattr(main, output_normalization)(y, output_name)
                except:
                    tf.compat.v1.logging.error(
                        'IGNNITION: The normalization function ' + str(
                            output_normalization) + ' is not defined in the main file.')
                    sys.exit(1)

            return x, y

        return x

    def input_fn_generator(self, filenames, shuffle=False, training=True, batch_size=1, data_samples=None):
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
            feature_list = self.model.get_all_features()
            adjacency_info = self.model.get_adjecency_info()
            interleave_list = self.model.get_interleave_tensors()
            interleave_sources = self.model.get_interleave_sources()
            output_name, output_normalization, _ = self.model.get_output_info()
            additional_input = self.model.get_additional_input_names()
            unique_additional_input = [a for a in additional_input if a not in feature_list]
            entity_names = self.model.get_entity_names()
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
                    ds = tf.data.Dataset.from_generator(self.generator.generate_from_dataset,
                                                        (types, tf.float32),
                                                        (shapes, tf.TensorShape(None)),
                                                        args=(
                                                            filenames, entity_names, feature_names, output_name, adjacency_info,
                                                            interleave_list,
                                                            unique_additional_input, training, shuffle))
                else:
                    data_samples = [json.dumps(t) for t in data_samples]
                    ds = tf.data.Dataset.from_generator(self.generator.generate_from_array,
                                                        (types, tf.float32),
                                                        (shapes, tf.TensorShape(None)),
                                                        args=(
                                                            data_samples, entity_names, feature_names, output_name,
                                                            adjacency_info,
                                                            interleave_list,
                                                            unique_additional_input, training, shuffle))

            else:
                if data_samples is None:
                    ds = tf.data.Dataset.from_generator(self.generator.generate_from_dataset,
                                                        (types),
                                                        (shapes),
                                                        args=(
                                                            filenames, entity_names, feature_names, output_name, adjacency_info,
                                                            interleave_list,
                                                            unique_additional_input, training, shuffle))
                else:
                    data_samples = [json.dumps(t) for t in data_samples]
                    ds = tf.data.Dataset.from_generator(self.generator.generate_from_array,
                                                        (types),
                                                        (shapes),
                                                        args=(
                                                            data_samples, entity_names, feature_names, output_name,
                                                            adjacency_info,
                                                            interleave_list,
                                                            unique_additional_input, training, shuffle))


            with tf.name_scope('normalization') as _:
                if training:
                    ds = ds.repeat()
                    ds = ds.map(lambda x, y: self.normalization(x, feature_list, output_name, output_normalization, y),
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)

                    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

                else:
                    ds = ds.map(lambda x: self.normalization(x, feature_list, output_name, output_normalization),
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
                    ds = tf.compat.v1.data.make_initializable_iterator(ds)

        return ds


    def r_squared(self, labels, predictions):
        """
        Parameters
        ----------
        labels:    tensor
            Label information
        predictions:    tensor
            Predictions of the model
        """
        total_error = tf.reduce_sum(tf.square(labels - tf.reduce_mean(labels)))
        unexplained_error = tf.reduce_sum(tf.square(labels - predictions))
        r_sq = 1.0 - tf.truediv(unexplained_error, total_error)

        m_r_sq, update_rsq_op = tf.compat.v1.metrics.mean(r_sq)

        return m_r_sq, update_rsq_op


    def model_fn(self, features, labels, mode):
        """
        Parameters
        ----------
        features:    dict
            All the features to be used as input
        labels:    tensor
            Tensor with the label information
        mode:    tensor
            Either train, eval or predict
        """

        # create the model
        gnn_model = Gnn_model(self.model)

        predictions = gnn_model(features, training=(mode == tf.estimator.ModeKeys.TRAIN))
        predictions = tf.squeeze(predictions)

        # prediction mode. Denormalization is done if so specified
        if mode == tf.estimator.ModeKeys.PREDICT:
            output_name, _, output_denorm = self.model.get_output_info()  # for now suppose we only have one output type
            try:
                predictions = getattr(main, output_denorm)(predictions, output_name)
            except:
                tf.compat.v1.logging.warn(
                    'IGNNITION: A denormalization function for output ' + output_name + ' was not defined. The output will be normalized.')

            return tf.estimator.EstimatorSpec(
                mode, predictions={
                    'predictions': predictions
                })

        loss_func_name = self.model.get_loss()
        # try to dynamically define the loss function from the keras documentation
        try:
            loss = globals()[loss_func_name]
            loss_function = loss()
            regularization_loss = sum(gnn_model.losses)
            loss = loss_function(labels, predictions)
            total_loss = loss + regularization_loss

            tf.summary.scalar('loss', loss)
            tf.summary.scalar('regularization_loss', regularization_loss)
            tf.summary.scalar('total_loss', total_loss)

        except:   # go to the main file and find the function by this name
            total_loss = getattr(main, loss_func_name)(predictions, labels, gnn_model)
            tf.summary.scalar('total_loss', total_loss)

        # evaluation mode
        if mode == tf.estimator.ModeKeys.EVAL:
            eval_metrics = {}

            # perform denormalization if defined
            output_name, _, output_denorm = self.model.get_output_info()

            if output_denorm != None:
                try:
                    labels_denormalized = getattr(main, output_denorm)(labels, output_name)
                    predictions_denormalized = getattr(main,output_denorm)(predictions, output_name)

                    label_mean_denorm = tf.keras.metrics.Mean()
                    _ = label_mean_denorm.update_state(labels_denormalized)
                    prediction_mean_denorm = tf.keras.metrics.Mean()
                    _ = prediction_mean_denorm.update_state(predictions_denormalized)
                    mae_denorm = tf.keras.metrics.MeanAbsoluteError()
                    _ = mae_denorm.update_state(labels_denormalized, predictions_denormalized)
                    mre_denorm = tf.keras.metrics.MeanRelativeError(normalizer= tf.abs(labels_denormalized))
                    _ = mre_denorm.update_state(labels_denormalized, predictions_denormalized)

                    eval_metrics['label_denorm/mean'] = label_mean_denorm
                    eval_metrics['prediction_denorm/mean'] = prediction_mean_denorm
                    eval_metrics['mae_denorm'] = mae_denorm
                    eval_metrics['mre_denorm'] = mre_denorm
                    eval_metrics['r-squared-denorm'] = self.r_squared(labels_denormalized, predictions_denormalized)

                except:
                    tf.compat.v1.logging.warn(
                        'IGNNITION: A denormalization function for output ' + output_name + ' was not defined. The output (and statistics) will use the normalized values.')

            # metrics calculations
            label_mean = tf.keras.metrics.Mean()
            _ = label_mean.update_state(labels)
            prediction_mean = tf.keras.metrics.Mean()
            _ = prediction_mean.update_state(predictions)
            mae = tf.keras.metrics.MeanAbsoluteError()
            _ = mae.update_state(labels, predictions)
            mre = tf.keras.metrics.MeanRelativeError(normalizer= tf.abs(labels))
            _ = mre.update_state(labels, predictions)

            eval_metrics['label/mean'] = label_mean
            eval_metrics['prediction/mean'] = prediction_mean
            eval_metrics['mae'] = mae
            eval_metrics['mre'] = mre
            eval_metrics['r-squared'] = self.r_squared(labels, predictions)


            return tf.estimator.EstimatorSpec(
                mode, loss=total_loss,
                eval_metric_ops= eval_metrics
            )

        assert mode == tf.estimator.ModeKeys.TRAIN

        grads = tf.gradients(total_loss, gnn_model.trainable_variables)

        summaries = [tf.summary.histogram(var.op.name, var) for var in gnn_model.trainable_variables]
        summaries += [tf.summary.histogram(g.op.name, g) for g in grads if g is not None]

        # dynamically define the optimizer
        optimizer_params = self.model.get_optimizer()
        op_type = optimizer_params['type']
        del optimizer_params['type']

        # dynamically define the adaptative learning rate if needed

        if 'schedule' in optimizer_params:
            schedule = optimizer_params['schedule']
            type = schedule['type']
            del schedule['type']  # so that only the parameters remain
            s = globals()[type]
            # create an instance of the schedule class indicated by the user. Accepts any schedule from keras documentation
            optimizer_params['learning_rate'] = s(**schedule)
            del optimizer_params['schedule']

        # create the optimizer
        o = globals()[op_type]
        optimizer = o(
            **optimizer_params)  # create an instance of the optimizer class indicated by the user. Accepts any loss function from keras documentation

        optimizer.iterations = tf.compat.v1.train.get_or_create_global_step()

        train_op = optimizer.apply_gradients(zip(grads, gnn_model.trainable_variables))

        logging_hook = tf.estimator.LoggingTensorHook(
            {"Loss": total_loss}
            , every_n_iter=10)

        return tf.estimator.EstimatorSpec(mode,
                                          loss=total_loss,
                                          train_op=train_op,
                                          training_hooks=[logging_hook]
                                          )

    def create_model(self):
        json_path = self.CONFIG['PATHS']['json_path']
        dimensions = self.find_dataset_dimensions(self.CONFIG['PATHS']['train_dataset'])
        model_info = Json_preprocessing(json_path, dimensions)  # read json
        return model_info


    def stream_read_json(self, f):
        start_pos = 1
        while True:
            try:
                obj = json.load(f)
                yield obj
                return
            except json.JSONDecodeError as e:
                f.seek(start_pos)
                json_str = f.read(e.pos)
                obj = json.loads(json_str)
                start_pos += e.pos +1
                a = f.read(1)
                if a == self.end_symbol:
                    yield obj
                    return
                yield obj


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
            tf.compat.v1.logging.error('IGNNITION: The file data.json was not found in ' + sample)
            sys.exit(1)

        try:
            file_samples = tar.extractfile('data.json')
            file_samples.read(1)
            aux = self.stream_read_json(file_samples)

            sample_data = next(aux) # read one single example #json.load(file_samples)[0]  # one single sample
            dimensions = {}
            for k, v in sample_data.items():
                # if it's a feature
                if not isinstance(v, dict) and isinstance(v, list):
                    if isinstance(v[0], str):
                        pass

                    # if it's a feature
                    elif isinstance(v[0], list):
                        dimensions[k] = len(v[0])
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

            return dimensions

        except:
            tf.compat.v1.logging.error('IGNNITION: Failed to read the data file ' + sample)
            sys.exit(1)


    def str_to_bool(self, a):
        """
        Parameters
        ----------
        a:    str
           Input
        """

        if a == 'True':
            return True
        else:
            return False


    def train_and_evaluate(self, training_samples = None, eval_samples = None):
        print()
        tf.compat.v1.logging.warn(
            'IGNNITION: Starting the training and evaluation process...\n---------------------------------------------------------------------------\n')

        filenames_train = self.CONFIG['PATHS']['train_dataset']
        filenames_eval = self.CONFIG['PATHS']['eval_dataset']

        model_dir = self.CONFIG['PATHS']['model_dir']
        if model_dir[-1] != '/':
            model_dir += '/'
        model_dir += 'CheckPoint'
        model_dir = model_dir + '/experiment_' + str(datetime.datetime.now())

        if self.CONFIG.has_option('PATHS', 'warm_start_path'):
            warm_start_setting = tf.estimator.WarmStartSettings(
                ckpt_to_initialize_from=self.CONFIG['PATHS']['warm_start_path'],
                vars_to_warm_start=["message_passing*", "model_initializations.*"])
        else:
            warm_start_setting = None

        d = {}
        if self.CONFIG.has_option('TRAINING_OPTIONS', 'execute_gpu'):
            if self.CONFIG['TRAINING_OPTIONS']['execute_gpu'] == 'False':
                d = {'GPU':0}

        my_checkpointing_config = tf.estimator.RunConfig(
            save_checkpoints_secs=int(self.CONFIG['TRAINING_OPTIONS']['save_checkpoints_secs']),
            keep_checkpoint_max=int(self.CONFIG['TRAINING_OPTIONS']['keep_checkpoint_max']),
            session_config=tf.compat.v1.ConfigProto(device_count=d)
        )

        estimator = tf.estimator.Estimator(
            model_fn=self.model_fn,
            model_dir=model_dir,
            warm_start_from=warm_start_setting,
            config=my_checkpointing_config
        )


        func_train = lambda: self.input_fn_generator(filenames_train,
                          shuffle=self.str_to_bool(self.CONFIG['TRAINING_OPTIONS']['shuffle_train_samples']),
                          batch_size=int(self.CONFIG['TRAINING_OPTIONS']['batch_size']), data_samples= training_samples)

        train_spec = tf.estimator.TrainSpec(
            input_fn= func_train,max_steps=int(self.CONFIG['TRAINING_OPTIONS']['train_steps']))


        func_eval = lambda: self.input_fn_generator(filenames_eval,
                      shuffle=self.str_to_bool(self.CONFIG['TRAINING_OPTIONS']['shuffle_train_samples']),
                      batch_size=int(self.CONFIG['TRAINING_OPTIONS']['batch_size']), data_samples=eval_samples)


        eval_spec = tf.estimator.EvalSpec(
            input_fn= func_eval,
                            throttle_secs=int(self.CONFIG['TRAINING_OPTIONS']['evaluation_time']),
                            steps=int(self.CONFIG['TRAINING_OPTIONS']['eval_samples']))

        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


    def predict(self, pred_samples = None):
        """
        Parameters
        ----------
        model_info:    object
         Object with the json information model
        """

        print()
        tf.compat.v1.logging.warn(
            'IGNNITION: Starting to make the predictions...\n---------------------------------------------------------\n')

        graph = tf.Graph()
        tf.compat.v1.disable_eager_execution()

        try:
            warm_path = self.CONFIG['PATHS']['warm_start_path']
        except:
            tf.compat.v1.logging.error(
                'IGNNITION: The path of the model to use for the predictions is unspecified. Please add a field warm_start_path in the train_options.ini with the corresponding path to the model you want to restore.')
            sys.exit(0)

        try:
            if pred_samples is None:
                data_path = self.CONFIG['PATHS']['predict_dataset']
            else:
                data_path = None
        except:
            tf.compat.v1.logging.error(
                'IGNNITION: The path of dataset to use for the prediction is unspecified. Please add a field predict_dataset in the train_config.ini file with the corresponding path to the dataset you want to predict.')
            sys.exit(0)

        with graph.as_default():
            gnn_model = Gnn_model(self.model)
            it = self.input_fn_generator(data_path, training=False, data_samples = pred_samples)
            features = it.get_next()

            # this predictions still need to be denormalized (or to do the predictions with the estimators)
            pred = gnn_model(features, training=False)

            # automatic denormalization
            output_name, _, output_denormalization = self.model.get_output_info()  # for now suppose we only have one output type
            try:
                pred = getattr(main, output_denormalization)(pred, output_name)
            except:
                tf.compat.v1.logging.warn('IGNNITION: A denormalization function for output ' + output_name + ' was not defined. The output will be normalized.')

        with tf.compat.v1.Session(graph=graph) as sess:
            sess.run(tf.compat.v1.local_variables_initializer())
            sess.run(tf.compat.v1.global_variables_initializer())
            saver = tf.compat.v1.train.Saver()

            # path to the checkpoint we want to restore
            saver.restore(sess, warm_path)

            all_predictions = []
            try:
                sess.run(it.initializer)
                while True:
                    p = sess.run([pred])
                    p = np.array(p)
                    p = p.flatten()
                    all_predictions.append(p)

            except tf.errors.OutOfRangeError:
                pass

            return all_predictions


    def scatter_nd_numpy(self, indices, updates, shape):
        target = np.zeros(shape, dtype=updates.dtype)
        indices = tuple(indices.reshape(-1, indices.shape[-1]).T)
        updates = updates.ravel()
        np.add.at(target, indices, updates)
        return target


    def get_k_best_accuracy(self):
        """
        Parameters
        ----------
        model_info:    object
         Object with the json information model
        """

        print()
        tf.compat.v1.logging.warn(
            'IGNNITION: Starting to make the predictions...\n---------------------------------------------------------\n')

        graph = tf.Graph()
        tf.compat.v1.disable_eager_execution()

        try:
            warm_path = self.CONFIG['PATHS']['warm_start_path']
        except:
            tf.compat.v1.logging.error(
                'IGNNITION: The path of the model to use for the predictions is unspecified. Please add a field warm_start_path in the train_options.ini with the corresponding path to the model you want to restore.')
            sys.exit(0)

        try:
            data_path = self.CONFIG['PATHS']['predict_dataset']
        except:
            tf.compat.v1.logging.error(
                'IGNNITION: The path of dataset to use for the prediction is unspecified. Please add a field predict_dataset in the train_config.ini file with the corresponding path to the dataset you want to predict.')
            sys.exit(0)

        with graph.as_default():
            model = Gnn_model(self.model)

            it = self.input_fn_generator(data_path,training=True)
            features, labels = it.get_next()    #normalized data
            # this predictions still need to be denormalized (or to do the predictions with the estimators)
            preds = model(features, training=False)

            # automatic denormalization
            output_name, _, output_denormalization = self.model.get_output_info()  # for now suppose we only have one output type
            #try:
            #    preds = eval(output_denormalization)(preds, output_name)
            #    labels = eval(output_denormalization)(labels, output_name)
            #except:
            #    tf.compat.v1.logging.warn('IGNNITION: A denormalization function for output ' + output_name + ' was not defined. The output will be normalized.')



        with tf.compat.v1.Session(graph=graph) as sess:
            sess.run(tf.compat.v1.local_variables_initializer())
            sess.run(tf.compat.v1.global_variables_initializer())
            saver = tf.compat.v1.train.Saver()

            # path to the checkpoint we want to restore
            saver.restore(sess, warm_path)

            try:
                accuracies_edges = []
                accuracies_paths = []
                accuracies_links = []

                for k in range(1, 100):
                    print("K = ", k)
                    sess.run(it.initializer)
                    edge_counter = 0
                    path_counter = 0
                    link_counter = 0
                    for j in range(100):
                        p, l, f = sess.run([preds, labels, features])
                        # take the best k positions (so we look for the best k links-paths)
                        p = np.array(p).flatten()
                        l = np.array(l)

                        p = (p * 0.3347904538999522) + 0.346676083526478
                        l = (l * 0.3347904538999522) + 0.346676083526478


                        k_best_predictions = np.argpartition(p, -k)[-k:]
                        k_best_labels = np.argpartition(l, -k)[-k:]

                        # BY SET (EDGES)
                        k_best_predictions_set = set(k_best_predictions.flatten())
                        k_best_labels_set = set(k_best_labels)

                        complementary = k_best_labels_set - k_best_predictions_set
                        intersection = k - len(complementary)
                        edge_counter += intersection / k


                        #OTHER ACCURACIES (OBTAIN MASKS)
                        links = f['src_adj_link_path']
                        paths = f['dst_adj_link_path']
                        indices = np.stack([paths, links], axis=1)
                        shape = [f['num_path'], f['num_link']]

                        mask_pred = self.scatter_nd_numpy(indices, p, shape)
                        mask_label = self.scatter_nd_numpy(indices, l, shape)

                        adj_matrix = self.scatter_nd_numpy(indices, np.ones_like(p), shape)

                        # ACCURACY TO FIND THE MOST IMPORTANT PATH
                        if k <= 182:
                            path_mean_pred = np.sum(mask_pred, axis=1) / np.sum(adj_matrix, axis=1)
                            path_mean_label = np.sum(mask_label, axis=1) / np.sum(adj_matrix, axis=1)

                            k_best_predictions = np.argpartition(path_mean_pred, -k)[-k:]
                            k_best_labels = np.argpartition(path_mean_label, -k)[-k:]

                            k_best_predictions = set(k_best_predictions.flatten())
                            k_best_labels = set(k_best_labels)

                            complementary = k_best_labels - k_best_predictions
                            intersection = k - len(complementary)
                            path_counter += intersection / k


                        # ACCURACY FINDING THE MOST IMPORTANT LINK
                        if k <= 42:
                            link_mean_pred = np.sum(mask_pred, axis=0) / np.sum(adj_matrix, axis=0)
                            link_mean_label = np.sum(mask_label, axis=0) / np.sum(adj_matrix, axis=0)

                            k_best_predictions = np.argpartition(link_mean_pred, -k)[-k:]
                            k_best_labels = np.argpartition(link_mean_label, -k)[-k:]

                            k_best_predictions = set(k_best_predictions.flatten())
                            k_best_labels = set(k_best_labels)

                            complementary = k_best_labels - k_best_predictions
                            intersection = k - len(complementary)
                            link_counter += intersection / k


                    edge_counter = edge_counter / 100
                    print("Edge accuracy: ", edge_counter)
                    accuracies_edges.append(edge_counter)

                    if k < 182:
                        path_counter = path_counter / 100
                        print("Path accuracy: ", path_counter)
                        accuracies_paths.append(path_counter)

                    if k < 42:
                        link_counter = link_counter / 100
                        print("Link accuracy: ", link_counter)
                        accuracies_links.append(link_counter)

                print("edge_accuracies:", accuracies_edges)
                print("link accuracies:", accuracies_links)
                print("edge_accuracies:", accuracies_paths)

            except tf.errors.OutOfRangeError:
                pass



    def computational_graph(self):
        """
        Parameters
        ----------
        model_description:    object
            Object with the json information model
        """

        print()
        tf.compat.v1.logging.warn(
            'IGNNITION: Generating the computational graph... \n---------------------------------------------------------\n')

        filenames_train = self.CONFIG['PATHS']['train_dataset']
        graph = tf.Graph()
        tf.compat.v1.disable_eager_execution()

        with graph.as_default():
            model = Gnn_model(self.model)
            it = self.input_fn_generator(filenames_train, training=False)
            data = it.get_next()
            pred= model(data, training=False)

        with tf.compat.v1.Session(graph=graph) as sess:
            sess.run(tf.compat.v1.local_variables_initializer())
            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(it.initializer)

        path = self.CONFIG['PATHS']['computational_graph_dir']
        if path[-1] != '/':
            path += '/'

        path += 'computational_graph'

        tf.compat.v1.summary.FileWriter(path, graph=sess.graph)
        tf.compat.v1.logging.warn('IGNNITION: The computational graph has been generated.')
