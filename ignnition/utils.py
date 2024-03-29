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

import re
import os
import sys
import json
import yaml
import tensorflow as tf

from ignnition.error_handling import YAMLNotFoundError, YAMLFormatError


class BColors:
    """
    Class which includes the hexadecimal code for a set of colors that are later used for printing messages
    """

    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    BLACK = '\033[00m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_failure(msg):
    """
    Prints a failure message

    Parameters
    ----------
    msg:    str
       Message to be printed
    """

    tf.print(BColors.FAIL + msg + BColors.ENDC, output_stream=sys.stderr)
    sys.exit(1)


def print_info(msg):
    """
    Prints an info message

    Parameters
    ----------
    msg:    str
       Message to be printed
    """

    tf.print(BColors.FAIL + msg + '\n' + BColors.ENDC, output_stream=sys.stderr)


def print_header(msg):
    """
    Prints a header message

    Parameters
    ----------
    msg:    str
       Message to be printed
    """

    tf.print(BColors.BOLD + msg + BColors.ENDC, output_stream=sys.stderr)


def stream_read_json(f):
    """
    It reads as a stream a dictionary with an array of json samples, and returns a generator that returns them eagerly.

    Parameters
    ----------
    f:
       Data
    """

    end_symbol = bytes(']', 'utf-8')
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
            start_pos += e.pos + 1
            a = f.read(1)
            if a == end_symbol:
                yield obj
                return
            yield obj


def str_to_bool(a):
    """
    It parses a string to boolean

    Parameters
    ----------
    a:    str
       Input
    """
    if a == 'True':
        return True
    else:
        return False


def save_global_variable(calculations, var_name, var_value):
    """
    Parameters
    ----------
    calculations: dict
        Dictionary with the current calculation of the GNN model indexed by name
    var_name:    String
        Name of the global variable to save
    var_value:    tensor
        Tensor value of the new global variable
    """
    calculations[var_name] = var_value


def get_global_variable(calculations, var_name):
    """
    Parameters
    ----------
    calculations: dict
        Dictionary with the current calculation of the GNN model indexed by name
    var_name:    String
        Name of the global variable to save
    """
    return calculations[var_name]


def get_global_var_or_input(calculations, var_name, f_):
    """
    Parameters
    ----------
    calculations: dict
        Dictionary with the current calculation of the GNN model indexed by name
    var_name:    str
        All the features to be used as input
    f_:    dict
        Input tensors of the sample
    """
    try:
        return get_global_variable(calculations, var_name)
    except KeyError:
        return f_[var_name]


def read_yaml(path, file_name=''):
    """
    Parameters
    ----------
    path:    str
        Path of the json file with the model description
    file_name: str
        Name of the file we aim to read
    """
    if os.path.isfile(path):
        with open(path, 'r') as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                message = re.sub(' +', ' ', str(exc).replace('\n', ' ')) + '.'
                raise YAMLFormatError(file=file_name, file_path=path, message=message[:1].upper() + message[1:])
    else:
        raise YAMLNotFoundError(file=file_name, file_path=path)
