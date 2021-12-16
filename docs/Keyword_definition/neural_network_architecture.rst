Step 4: Neural Network architectures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this section, we define the architecture of the neural networks that we referenced in all the previous sections. For
this, we just need to define an array of :ref:`Neural Network object <neural_network_object>`. Note that we will use
the very same syntax to define either *Feed-forward NN* or *Recurrent NN*. Let us describe what a
:ref:`Neural Network object <neural_network_object>` looks like:

.. _neural_network_object:

Neural Network object
~~~~~~~~~~~~~~~~~~~~~

A Neural Network object refers to the architecture of a specific Neural Network. To do so, we must define two main
fields, these being *nn_name* and *nn_architecture* which we define below.

We can now review in more depth each of its available parameters:


.. contents::
    :local:
    :depth: 1

----

Parameter: nn_name
""""""""""""""""""

**Description:** Name of the Neural Network.

**Accepted values:** String. This name must match all the references to this Neural Network from all the previous
sections (e.g., the name of the *NN* of the previous example would be *my_neural_network*)

----

Parameter: nn_architecture
""""""""""""""""""""""""""

**Description:** Definition of the actual architecture of the *NN*.

**Accepted values:** Array of Layer objects (e.g., a single *Dense* layer for the previous *NN*)

Let us now, for sake of the explanation, provide a simple example of how a *Neural Network* object can be defined:

.. code-block:: yaml

   neural_networks:
   - nn_name: my_neural_network
     nn_architecture:
     - type_layer: Dense
       units: readout_units

Layer object
~~~~~~~~~~~~

To define a Layer, we rely greatly on the well-known `tf.keras library <https://www.tensorflow.org/api_docs/python/tf/keras/layers>`_.
In consequence, we just require the user to define the following field.

----

Parameter: type_layer
"""""""""""""""""""""

**Description:** Here we must indicate the type of layer to be used. Please write only the layers accepted by the
`tf.keras.layers library <https://www.tensorflow.org/api_docs/python/tf/keras/layers>`_ using the same syntax.

**Allowed values:** String. It must match a layer from the *tf.keras.layers library*

.. code-block:: yaml

   - type_layer: Dense/Softmax/...
     ...

Other parameters
""""""""""""""""

Additionally, the user can define any other parameter from the `tf.keras library <https://www.tensorflow.org/api_docs/python/tf/keras/layers>`_
corresponding to the type of layer defined. Note that on many occasions, the user is required to define layer-specific attributes (e.g., the number of units when creating a Dense layer). Thus, please make sure to define all
mandatory parameters, and then, additionally, define optional parameters if needed.

E.g., if we define a Dense layer, we must first define the required parameter *units* (as specified by Tensorflow).
Then, we can also define any optional parameter for the Dense class (visit `documentation <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense>`_)
such as the activation or the use of bias.

.. code-block:: yaml

   - type_layer: Dense
     units: 32
     activation: relu
     use_bias: False
