import tensorflow as tf
import tensorflow.keras.activations
from keras import backend as K
import sys
from ignnition.utils import *


class Custom_layer:
    """
    This class implements a custom layer, which represents  tf.keras.layers object with the parameters/attributes specified by the user.

    Attributes
    ----------
    type:    str
        Type of layer to be created (following the tensorflow documentation)
    parameters: dict
        Dictionary with the parameters to be applied to the new layer (following the tf docu).

    Methods:
    ----------
    __prepocess_parameters(self)
       Parses several parameters which are in string type to its corresponding type, this being integer, or other tensorflow objects accordingly.
    """

    def __init__(self, type, parameters):
        """
        Parameters
        ----------
        type:    str
            Type of layer to be created (following the tensorflow documentation)
        parameters: dict
            Dictionary with the parameters to be applied to the new layer (following the tf docu).
        """

        self.type = type
        if 'type_layer' in parameters:
            del parameters['type_layer']
        self.parameters = parameters
        self.__prepocess_parameters()

    def __prepocess_parameters(self):
        for k, v in self.parameters.items():
            if v == 'None':
                self.parameters[k] = None

            elif v == 'True':
                self.parameters[k] = True

            elif v == 'False':
                self.parameters[k] = False

            elif 'regularizer' in k:
                try:
                    self.parameters[k] = tf.keras.regularizers.l2(float(self.parameters.get(k)))
                except:
                    print_failure("The " + k + " parameter '" + str(self.parameters.get(
                        k)) + "' in layer of type " + self.type + " is invalid. Please make sure it is a numerical value.")


            elif 'activation' in k:  # already ensures that it was not None
                try:
                    self.parameters['activation'] = getattr(tf.nn, v)
                except:
                    print_failure(
                        "The activation '" + v + "' is not a valid function from the tf.nn library. Please check the documentation and the spelling of the function.")


class Recurrent_Update_Cell(Custom_layer):
    """
    Class that represents an RNN model used only for the update. The key difference with the other recurrent possibility is that in the update we need to explicitely pass the initial state, while in other stages of the model we don't need to do so.
    Thus, we keep this a single layer instead of incorporating it to a Sequential model.

    Methods:
    --------
    get_tensorflow_object(self, destination_dimension
        Returns a tensorflow object with of this recurrent type with the destination_dimension as the number of units

    perform_unsorted_update(self, model, src_input, old_state)
        Updates the hidden state using the result of applying an input to the model obtained (order doesn't matter)

    perform_sorted_update(self,model, src_input, dst_name, old_state, final_len, num_dst )
        Updates the hidden state using the result of applying an input to the model obtained (order matters)

    """

    def __init__(self, type, parameters):
        """
        Parameters
        ----------
        type:    str
            Type of layer to be created (following the tensorflow documentation)
        parameters: dict
            Dictionary with the parameters to be applied to the new layer (following the tf docu).
        """

        super(Recurrent_Update_Cell, self).__init__(type=type, parameters=parameters)

    def get_tensorflow_object(self, dst_dim):
        """
        Parameters
        ----------
        dst_dim:    int
            Dimension of the destination nodes. Thus, number of units of the RNN model
        """

        self.parameters['units'] = dst_dim
        try:
            c_ = getattr(tf.keras.layers, self.type + 'Cell')
        except:
            print_failure(
                "Error when trying to define a RNN of type '" + self.type + "' since this type does not exist. Check the valid RNN cells that Keras allow to define.")

        try:
            layer = c_(**self.parameters)
        except:
            print_failure(
                "Error when creating the RNN of type '" + self.type + "' since invalid parameters were passed. Check the documentation to check which parameters are acceptable or check the spelling of the parameters' names.")

        return layer

    def perform_unsorted_update(self, model, src_input, old_state, dst_dim):
        """
        Parameters
        ----------
        model:    object
            Update model
        src_input:  tensor
            Input for the update operation
        old_state:  tensor
            Old hs of the destination entity
        dst_dim: int
            Dimension of the destination nodes
        """
        src_input = tf.ensure_shape(src_input, [None, dst_dim])
        new_state, _ = model(src_input, [old_state])
        return new_state

    def perform_sorted_update(self, model, src_input, dst_name, old_state, final_len):
        """
        Parameters
        ----------
        model:    object
            Update model used for the update
        src_input:  tensor
            Input for the update operation
        dst_name:   str
            Destination entity name
        old_state:  tensor
            Old hs of the destination entity's nodes
        final_len:  tensor
            Number of source nodes for each destination
        """

        rnn = tf.keras.layers.RNN(model, name=str(dst_name) + '_update')
        final_len.set_shape([None])
        new_state = rnn(inputs=src_input, initial_state=old_state, mask=tf.sequence_mask(final_len))
        return new_state


class Feed_forward_Layer(Custom_layer):
    """
    Class that represents a layer of a feed_forward neural network

    Methods:
    --------
    get_tensorflow_object(self)
        Returns a tensorflow object of the containing layer

    get_tensorflow_object_last(self, destination_units)
        Returns a tensorflow object of the last layer of the model, and sets its previous layer and the number of output units for it to have.
    """

    def __init__(self, type, parameters):
        """
        Parameters
        ----------
        type:    str
            Type of layer to be created (following the tensorflow documentation)
        parameters: dict
            Dictionary with the parameters to be applied to the new layer (following the tf docu).
        """

        super(Feed_forward_Layer, self).__init__(type=type, parameters=parameters)

    def get_tensorflow_object(self):
        try:
            c_ = getattr(tf.keras.layers, self.type)
        except:
            print_failure(
                "The layer of type '" + self.type + "' is not a valid tf.keras layer. Please check the documentation to write the correct way to define this layer. ")

        try:
            layer = c_(**self.parameters)
        except:
            parameters_string = ''
            for k, v in self.parameters.items():
                parameters_string += k + ': ' + v + '\n'
            print_failure("One of the parameters passed to the layer of type '" + self.type + "' is incorrect. \n " +
                          "You have defined the following parameters: \n" + parameters_string)

        return layer

    def get_tensorflow_object_last(self, dst_units):
        """
        Parameters
        ----------
        dst_units:    int
            Number of units that the recurrent cell will have
        """
        try:
            c_ = getattr(tf.keras.layers, self.type)
        except:
            print_failure(
                "The layer of type '" + self.type + "' is not a valid tf.keras layer. Please check the documentation to write the correct way to define this layer. ")

        self.parameters['units'] = dst_units  # can we assume that it will always be units??

        try:
            layer = c_(**self.parameters)
        except:
            parameters_string = ''
            for k, v in self.parameters.items():
                parameters_string += k + ': ' + v + '\n'
            print_failure("One of the parameters passed to the layer of type '" + self.type + "' is incorrect. \n " +
                          "You have defined the following parameters: \n" + parameters_string)

        return layer


class Feed_forward_model:
    """
    Class that represents a feed_forward neural network model

    Attributes:
    ----------
    layers:    array
        Layers contained in this feed-forward

    Methods:
    --------
    __is_recurrent(self, type)
        Given the definition of a layer, it returns True if this is a recurrent layer (using GRU or LSTM)
    construct_tf_model(self, var_name, input_dim, dst_dim = None, is_readout = False, dst_name = None)
        Returns the corresponding neural network object
    """

    def __init__(self, model, model_role):
        """
        Parameters
        ----------
        model:    dict
            Information regarding the architecture of the feed-forward
        model_role: str
            Defines the role of the feed_forward. Only useful to save the variables and for debugging purposes
        """

        self.layers = []
        counter = 0
        # create the layers
        if 'architecture' in model:
            dict = model['architecture']

            for l in dict:
                type_layer = l['type_layer']  # type of layer

                # if this is a RNN, we need to do reshaping of the model
                if self.__is_recurrent(type_layer):
                    reshape_layer = Feed_forward_Layer('Reshape', {
                        "target_shape": (1, -1)})  # this should take as size the previous shape
                    self.layers.append(reshape_layer)

                if 'name' not in l:
                    l['name'] = 'layer_' + str(counter) + '_' + type_layer + '_' + str(model_role)
                # del l['type_layer']  # leave only the parameters of the layer

                layer = Feed_forward_Layer(type_layer, l)
                self.layers.append(layer)
                counter += 1

    def __is_recurrent(self, type):
        """
        Parameters
        ----------
        type:    str
            Type of the layer that we are considering
        """
        return True if (type == 'LSTM' or type == 'GRU') else False

    def construct_tf_model(self, var_name, input_dim, dst_dim=None, is_readout=False, dst_name=None):
        """
        Parameters
        ----------
        var_name:    str
            Name of the variables
        input_dim:  int
            Dimension of the input of the model
        dst_dim:  int
            Dimension of the destination hs if any
        is_readout: bool
            Is a model used for the readout?
        dst_name:   str
            Name of the destination entity
        """

        model = tf.keras.models.Sequential()
        model.add(tf.keras.Input(shape=(input_dim)))

        layer_counter = 1
        n = len(self.layers)

        for j in range(n):
            current_layer = self.layers[j]
            try:
                # if it's the last layer and we have defined an output dimension
                if j == (n - 1) and dst_dim is not None:
                    layer_model = current_layer.get_tensorflow_object_last(dst_dim)
                else:
                    layer_model = current_layer.get_tensorflow_object()

                model.add(layer_model)

            except:
                if dst_dim is None:
                    if is_readout:
                        print_failure('The layer ' + str(
                            layer_counter) + ' of the readout is not correctly defined. Check keras documentation to make sure all the parameters are correct.')
                    else:
                        print_failure('The layer ' + str(
                            layer_counter) + ' of the message creation neural network in the message passing to ' + str(
                            dst_name) + ' is not correctly defined. Check keras documentation to make sure all the parameters are correct.')

                else:
                    print_failure('The layer ' + str(
                        layer_counter) + ' of the update neural network in message passing to ' + str(dst_name) +
                                  ' is not correctly defined. Check keras documentation to make sure all the parameters are correct.')

            layer_counter += 1
        output_shape = model.output_shape[-1]
        return [model, output_shape]
