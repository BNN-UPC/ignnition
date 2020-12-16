import tensorflow as tf
import tensorflow.keras.activations
from keras import backend as K
import sys
from ignnition.utils import *

class Recurrent_Cell:
    """
    Class that represents an RNN model

    Attributes
    ----------
    type:    str
        Type of recurrent model to be used
    params:     dict
        Additional parameters

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
            Type of recurrent model to be used
        parameters:    dict
           Additional parameters of the model
        """
        self.type = type
        self.parameters = parameters
        self.trainable = self.parameters.get('trainable', 'True')
        self.trainable = 'True' == self.trainable

        for k, v in self.parameters.items():
            if v == 'None':
                self.parameters[k] = None

    def get_tensorflow_object(self, destination_dimension):
        """
        Parameters
        ----------
        destination_dimension:    int
            Number of units that the recurrent cell will have
        """
        self.parameters['units'] = destination_dimension
        try:
            c_ = getattr(tf.keras.layers, self.type + 'Cell')
        except:
            print_failure("Error when trying to define a RNN of type '" + self.type + "' since this type does not exist. Check the valid RNN cells that Keras allow to define.")

        try:
            layer = c_(**self.parameters)
            layer.trainable = self.trainable
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
        """
        src_input = tf.ensure_shape(src_input, [None, dst_dim])
        new_state, _ = model(src_input, [old_state])
        return new_state

    def perform_sorted_update(self,model, src_input, dst_name, old_state, final_len):
        """
        Parameters
        ----------
        model:    object
            Update model
        src_input:  tensor
            Input for the update operation
        dst_name:   str
            Destination entity name
        old_state:  tensor
            Old hs of the destination entity
        final_len:  tensor
            Number of source nodes for each destination
        num_dst:    int
            Number of destination nodes
        """
        gru_rnn = tf.keras.layers.RNN(model, name=str(dst_name) + '_update')
        final_len.set_shape([None])
        new_state = gru_rnn(inputs = src_input, initial_state = old_state, mask=tf.sequence_mask(final_len))
        return new_state


class Feed_forward_Layer:
    """
    Class that represents a layer of a feed_forward neural network

    Attributes
    ----------
    type:    str
        Type of recurrent model to be used
    params:     dict
        Additional parameters


    Methods:
    --------
    get_tensorflow_object(self, l_previous)
        Returns a tensorflow object of the containing layer, and sets its previous layer.

    get_tensorflow_object_last(self, l_previous, destination_units)
        Returns a tensorflow object of the last layer of the model, and sets its previous layer and the number of output units for it to have.

    """
    def __init__(self, type, parameters):
        """
        Parameters
        ----------
        type:    str
            ?
        parameters:    dict
            Additional parameters of the model
        """
        self.type = type

        if 'kernel_regularizer' in parameters:
            try:
                parameters['kernel_regularizer'] = tf.keras.regularizers.l2(float(parameters.get('kernel_regularizer')))
            except:
                print_failure("The kernel regularizer parameter '" + str(parameters.get('kernel_regularizer')) + "' in layer of type " + self.type + " is invalid. Please make sure it is a numerical value.")

        if 'activation' in parameters:
            activation = parameters.get('activation')
            if activation == 'None':
                parameters['activation'] = None
            else:
                try:
                    parameters['activation'] = getattr(tf.nn, activation)
                except:
                    print_failure("The activation '" + activation + "' is not a valid function from the tf.nn library. Please check the documentation and the spelling of the function.")


        self.trainable = parameters.get('trainable', 'True')
        parameters['trainable'] = 'True' == self.trainable
        self.parameters = parameters

    def get_tensorflow_object(self, l_previous):
        """
        Parameters
        ----------
        l_previous:    object
            Previous layer of the architecture
        """
        try:
            c_ = getattr(tf.keras.layers, self.type)
        except:
            print_failure("The layer of type '" + self.type + "' is not a valid tf.keras layer. Please check the documentation to write the correct way to define this layer. ")

        try:
            layer = c_(**self.parameters)(l_previous)
        except:
            parameters_string = ''
            for k,v in self.parameters.items():
                parameters_string += k + ': ' + v + '\n'
            print_failure("One of the parameters passed to the layer of type '" + self.type +  "' is incorrect. \n " +
                        "You have defined the following parameters: \n" + parameters_string)

        return layer


    def get_tensorflow_object_last(self, l_previous, destination_units):
        """
        Parameters
        ----------
        l_previous:    object
            Previous layer of the architecture
        destination_dimension:    int
            Number of units that the recurrent cell will have
        """
        try:
            c_ = getattr(tf.keras.layers, self.type)
        except:
            print_failure("The layer of type '" + self.type + "' is not a valid tf.keras layer. Please check the documentation to write the correct way to define this layer. ")

        self.parameters['units'] = destination_units

        try:
            layer = c_(**self.parameters)(l_previous)
        except:
            parameters_string = ''
            for k,v in self.parameters.items():
                parameters_string += k + ': ' + v + '\n'
            print_failure("One of the parameters passed to the layer of type '" + self.type +  "' is incorrect. \n " +
                        "You have defined the following parameters: \n" + parameters_string)

        return layer


class Feed_forward_model:
    """
    Class that represents a feed_forward neural network

    Attributes:
    ----------
    layers:    array
        Layers contained in this feed-forward
    counter:    int
        Counts the current number of layers


    Methods:
    --------
    construct_tf_model(self, var_name, input_dim, dst_dim = None, is_readout = False, dst_name = None)
        Returns the corresponding neural network object

    add_layer(self, **l)
        Add a layer using a dictionary as input

    add_layer_aux(self, l)
        Add a layer

    """

    def __init__(self, model, model_role, n_extra_params = 0):
        """
        Parameters
        ----------
        model:    dict
            Information regarding the architecture of the feed-forward
        """

        self.layers = []
        self.counter = 0
        self.extra_params = n_extra_params

        if 'architecture' in model:
            dict = model['architecture']
            for l in dict:
                type_layer = l['type_layer']  # type of layer
                if 'name' not in l:
                    l['name'] = 'layer_' + str(self.counter) + '_' + type_layer + '_' + str(model_role)
                del l['type_layer']  # leave only the parameters of the layer

                layer = Feed_forward_Layer(type_layer, l)
                self.layers.append(layer)
                self.counter += 1


    def construct_tf_model(self, var_name, input_dim, dst_dim = None, is_readout = False, dst_name = None):
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
            Is readout?
        dst_name:   str
            Name of the destination entity
        """
        setattr(self, str(var_name) + "_layer_" + str(0),
                tf.keras.Input(shape=(input_dim)))

        layer_counter = 1
        n = len(self.layers)

        for j in range(n):
            l = self.layers[j]
            l_previous = getattr(self, str(var_name) + "_layer_" + str(layer_counter - 1))
            try:
                # if it's the last layer and we haven't defined an output dimension
                if j==(n-1) and dst_dim is not None:
                    layer_model = l.get_tensorflow_object_last(l_previous, dst_dim)
                else:
                    layer_model = l.get_tensorflow_object(l_previous)

                setattr(self, str(var_name) + "_layer_" + str(layer_counter), layer_model)

            except:
                if dst_dim is None: #message_creation
                    if is_readout:
                        print_failure('The layer ' + str(
                                layer_counter) + ' of the readout is not correctly defined. Check keras documentation to make sure all the parameters are correct.')
                    else:
                        print_failure('The layer ' + str(
                            layer_counter) + ' of the message creation neural network in the message passing to ' + str(
                            dst_name) +' is not correctly defined. Check keras documentation to make sure all the parameters are correct.')

                else:
                    print_failure('The layer ' + str(
                            layer_counter) + ' of the update neural network in message passing to ' + str(dst_name) +
                        ' is not correctly defined. Check keras documentation to make sure all the parameters are correct.')

            output_shape = int(layer_model.shape[1])
            layer_counter += 1

        model = tf.keras.Model(inputs=getattr(self, str(var_name) + "_layer_" + str(0)),
                               outputs=getattr(self, str(var_name) + "_layer_" + str(
                                   layer_counter - 1)))
        return [model, output_shape]


