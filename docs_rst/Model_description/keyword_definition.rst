.. _keyword-definition:

Keyword definition
------------------

In this section we will focus in more depth on what are the keywords available to design each of the sections that
themselves define the GNN, and how to use them. More specifically, we will cover the keywords for each of the following
sections.


.. contents::
    :local:
    :depth: 1

For more specific details regarding the operations and source entity objects available, please check the sections
:ref:`Source entity object <entity_object>` and :ref:`Operation objects<operation-object>`


Step 1: Entity definition
^^^^^^^^^^^^^^^^^^^^^^^^^

In order to create the entities, we must define a list "entities". For this, we must define an object "Entity".
We shall now describe the different keywords that the user must / can define to model the new entity, these being:


.. contents::
    :local:
    :depth: 1


----

Parameter: name
~~~~~~~~~~~~~~~

**Description:** Name that we assign to the new entity. This name is important as we will use it from now on to reference the nodes that belong to this entity.

**Accepted values:** String of the choice of the user.

E.g., below we show how we would define an entity of name *entity1*.

.. code-block:: yaml

   name: entity1

----

Parameter: state_dim
~~~~~~~~~~~~~~~~~~~~

**Description:** Dimension of the hidden states of the nodes of this entity.

**Accepted values:** Natural number

.. code-block:: yaml

   state_dim: 32

----

Parameter: initial_state
~~~~~~~~~~~~~~~~~~~~~~~~

**Description:** Array of Operation object defining incrementally the initial_state.

**Accepted values:** Array of :ref:`Operation objects <operation-object>`.

Step 2: Message-passing phase
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now we define the keywords that the user can use to design the message passing phase of the present *GNN*. To do so, we will cover the following keywords:

.. contents::
    :local:
    :depth: 1

Parameter: num_iterations
~~~~~~~~~~~~~~~~~~~~~~~~~

**Description:** Number of times that all the stages must be repeated (iteration of the message passing phase).

**Accepted values:** Natural number (Normally between 3 and 8)

.. code-block:: yaml

   num_iterations: 8

----

Parameter: stages
~~~~~~~~~~~~~~~~~

**Description:** Stages is the array of stage object which ultimately define all the parts of the message passing.

**Accepted values:** Array of [Stage objects](#stage, each of which represents a time-step of the algorithm.

Stage:
""""""

To define a stage, we must define all the single message passing that take place during that stage (a given time-step
of the algorithm). This is to define all the single message-passing which define how potentially many entities send
messages to a destination entity.

Parameter: stage_message_passings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Description:** Contains the single message-passings (the process of one entity nodes sending messages to another one),
which we assign to this stage (time-step of the algorithm)

**Accepted values:** Array of :ref:`Single message-passing objects <single-message-passing>`.

.. _single-message-passing:

Single message-passing:
~~~~~~~~~~~~~~~~~~~~~~~

This object defines how the nodes of potentially many entity types send messages simultaneously to the nodes of a
given destination entity. To do so, we must define the following parameters:


.. contents::
    :local:
    :depth: 1


Parameter: destination entity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Description:** Name of the destination entity of this single message-passing. In other words, the entity nodes
receiving the messages.

**Accepted values:** String. It must match the name of an entity previously defined (see :ref:`entity name <entity_name>`).

.. code-block:: yaml

   destination_entity: my_dst_entity

----

Parameter: source_entities
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Description:** Array of the source entities sending messages to the destination entity (defined before) in this
single message-passing. This is, all these sending entities will send messages simultaneously to the defined
destination entity.

**Accepted values:** Array of :ref:`Source entity objects <entity_object>`.

----

Parameter: aggregation
~~~~~~~~~~~~~~~~~~~~~~

**Description:** Defines the aggregation function, which will take as input all the messages received by each of the
destination nodes respectively, and aggregates them together into a single representation. Note that, to define
potentially very complex function, we define this as a pipeline of aggregation operations

**Accepted values:** Array of :ref:`Aggregation operation <aggregation_operation>`.

----

Parameter: update
~~~~~~~~~~~~~~~~~

**Description:** Defines the update function. This function will be applied to each of the destination nodes, and
given the aggregated input and the current hidden state, will produce the updated hidden-state.

**Accepted values:** :ref:`Update operation <update-operation>`.

Step 3: Readout
^^^^^^^^^^^^^^^

Just as for the case of the message function, the readout function can potentially be very complex. For this, we
follow a similar approach. We define the readout as a pipe-line of :ref:`Operation object <operation-object>` which
shall allow us to define very complex functions. Again, each of the operations will keep the field *output_name*
indicating the name with which we can reference/use the result of this operation in successive operations.

The main particularity for the definition of the readout is that in one of the operations (normally the last one),
will have to include the name of the *output_label* that we aim to predict. To do so, include the keyword presented
below as a property of last *Operation* of your readout function (the output of which will be used as output of
the *GNN*\ ).

Another important consideration is that in this case, the user can use *entity1_initial_state* as part of the input
of an operation (where *entity1* can be replaced for any entity name of the model). With this, the operation will take
as input the initial hidden states that were initialized at the beginning of the execution, and thus, before the
message-passing phase.

Parameter: output_label
~~~~~~~~~~~~~~~~~~~~~~~

**Description:** Name referencing the labels that we want to predict, which must be defined in the dataset.

**Allowed values:** Array of strings. The names should match the labels specified in the dataset.

Let us see this with a brief example of a simple readout function based on two
:ref:`Neural Network operations <neural-network-operation>`. In this case we apply two neural networks which are
intially to each of the nodes of type *entity1*. Then, the output is concatenated together with each of the nodes of
type *entity2* (as long that there is the same number of nodes of each entity) and then applied to the second neural
network *my_network2*. Note that the last operation includes the definition of *my_label*, which is the name of the
label found in the dataset. To specify this label, we write *$my_label* so as to indicate that this keywords refers to
data that *IGNNITION* can find in the corresponding dataset.

.. code-block:: yaml

   readout:
   - type: neural_network
     input: [entity1]
     nn_name: my_network1
     output_label: output1
   - type: neural_network
     input: [output1, entity2]
     nn_name: my_network2
     output_label: [$my_label]

Notice, however, that *output_label* may contain more than one label. For instance, consider the case in which we
want that the readout function predicts two properties of a node, namely *label1* and *label2*. For simplicity, let us
consider these labels to be single values --even though the same procedure applies when they represent 1-d arrays. For
this, we make the following adaptations of the previous model:

.. code-block:: yaml

   readout:
   - type: neural_network
     input: [entity1]
     nn_name: my_network1
     output_label: output1
   - type: neural_network
     input: [output1, entity2]
     nn_name: my_network2
     output_label: [$label1, $label2]

In this case, hence, *my_network2* will output two predictions, one for each of the target labels. Then, *IGNNITION*
will internally process this and backpropagate accordingly, so as to force the GNN to learn to predict both properties,
simultaneously.

Step 4: Neural Network architectures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this section we define the architecture of the neural networks that we referenced in all the previous sections. For
this, we just need to define an array of :ref:`Neural Network object <neural_network_object>`. Note that we will use
the very same syntax to define either *Feed-forward NN* or *Recurrent NN*. Let us describe what a
:ref:`Neural Network object <neural_network_object>` looks like:

.. _neural_network_object:

Neural Network object
~~~~~~~~~~~~~~~~~~~~~

A Neural Network object refers to the architecture of an specific Neural Network. To do so, we must define two main
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
corresponding to the type of layer defined. Note that in many occasions, the user is in fact required to define layer
specific attributes (e.g., the number of units when creating a Dense layers). Thus, please make sure to define all
mandatory parameters, and then, additionally, define optional parameters if needed.

E.g., if we define a Dense layer, we must first define the required parameter *units* (as specified by Tensorflow).
Then, we can also define any optional parameter for the Dense class (visit `documentation <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense>`_)
such as the activation or the use of bias.

.. code-block:: yaml

   - type_layer: Dense
     units: 32
     activation: relu
     use_bias: False