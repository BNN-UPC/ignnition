import json
import tensorflow as tf
import sys
import os
from collections import OrderedDict


class K_best(tf.keras.callbacks.Callback):
    """
    A subclass of the callback preset class which implements the functionality to keep only the best k checkpoints of the execution (instead of the best one implemented in Tf).

    Attributes
    ----------
    output_path:    str
        Path where the checkpoints are saved
    file_loss: dict
        Dictionary where we keep the loss of each of the checkpoints saved
    k: int
        Number of checkpoints to keep

    Methods:
    ----------
    on_epoch_end(self, src_input, indices)
       At the end of each epoch, we check which of the checkpoints we need to delete (if any)
    """

    def __init__(self, output_path, k=None):
        """
        Parameters
        ----------
        output_path:    str
            Path where the checkpoints are saved
        k: int
            Number of checkpoints to keep
        """
        self.output_path = output_path
        self.files_loss = {}
        self.k = k

    def on_epoch_end(self, epoch, logs={}):
        """
        Parameters
        ----------
        epoch:    int
            Epoch number
        logs:    dict
            Dictionary with the information of the current epoch
        """

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
