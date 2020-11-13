import json
import tensorflow as tf
import sys
import os
from collections import OrderedDict


class Custom_progressbar(tf.keras.callbacks.Callback):
    def __init__(self, model_dir, num_epochs, mini_epoch_size, metric_names, k=None):
        self.model_dir = model_dir
        self.files_loss = {}
        self.k = k
        self.epoch = 0
        self.metric_names = metric_names
        self.num_epochs = num_epochs
        self.mini_epoch_size = mini_epoch_size
        self.num_samples = 0

    def on_epoch_begin(self, epoch, logs):
        self.epoch = epoch + 1
        print("\nEpoch {}/{}".format(self.epoch, self.num_epochs))
        self.progBar = tf.keras.utils.Progbar(self.mini_epoch_size, stateful_metrics=self.metric_names)

    def on_train_batch_end(self, batch_id, logs=None):
        self.num_samples += 1
        logs['sample_num'] = self.num_samples
        self.progBar.update(batch_id, values=logs.items())

    def on_test_end(self, logs=None):
        logs = [('val_' + k, v) for k, v in logs.items()]
        logs.append(('sample_num', self.num_samples))
        self.progBar.update(self.mini_epoch_size, values=logs)

    def on_epoch_end(self, epoch, logs={}):
        # if we are aiming to save only the k best models
        if self.k is not None:
            loss = logs["loss"]
            name = "weights." + str("{:02d}".format(self.epoch)) + '-' + str("{:.2f}".format(loss)) + '.hdf5'
            self.files_loss[name] = loss

            if len(self.files_loss) >= self.k:
                # sort by value in decreasing order
                d_descending = OrderedDict(sorted(self.files_loss.items(), key=lambda kv: kv[1], reverse=True))
                n = len(d_descending)

                # delete the len(d_descending - k) first files
                num_deletions = n - self.k
                file_delete = list(d_descending.items())[0:num_deletions]
                for name, _ in file_delete:
                    path = self.model_dir + '/ckpt/' + name
                    os.remove(path)
                    del self.files_loss[name]
