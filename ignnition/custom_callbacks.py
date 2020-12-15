import json
import tensorflow as tf
import sys
import os
from collections import OrderedDict

class Custom_progressbar(tf.keras.callbacks.Callback):
    def __init__(self, output_path, num_epochs, mini_epoch_size, metric_names, k=None):
        self.output_path = output_path
        self.files_loss = {}
        self.k = k

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
                    path = self.output_path + '/ckpt/' + name
                    os.remove(path)
                    del self.files_loss[name]
