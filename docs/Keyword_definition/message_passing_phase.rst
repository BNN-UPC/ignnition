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

**Description:** Stages define the array of stage objects which ultimately define all the parts of the message passing.

**Accepted values:** Array of [Stage objects](#stage, each of which represents a time-step of the algorithm.

Stage:
""""""

To define a stage, we must define all the single message passing that take place during that stage (a given time-step
of the algorithm). This is to define all the single message-passing which defines how potentially many entities send
messages to a destination entity.

Parameter: stage_message_passings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Description:** Contains the single message-passings (the process of one entity nodes sending messages to another one),
which we assign to this stage (time-step of the algorithm)

**Accepted values:** Array of :ref:`Single message-passing objects <single-message-passing>`.

.. _single-message-passing:

Single message-passing:
"""""""""""""""""""""""

This object defines how the nodes of potentially many entity types send messages simultaneously to the nodes of a
given destination entity. To do so, we must define the following parameters:


.. contents::
    :local:
    :depth: 1

Parameter: source_entities
""""""""""""""""""""""""""

**Description:** Array of the source entities sending messages to the destination entity (defined before) in this
single message-passing. This is, all these sending entities will send messages simultaneously to the defined
destination entity.

**Accepted values:** Array of :ref:`Source entity objects <entity_object>`.

----

Parameter: destination_entity
"""""""""""""""""""""""""""""

**Description:** Name of the destination entity of this single message-passing. In other words, the entity nodes
receiving the messages.

**Accepted values:** String. It must match the name of an entity previously defined (see :ref:`entity name <entity_name>`).

.. code-block:: yaml

   destination_entity: my_dst_entity

----

Parameter: aggregation
""""""""""""""""""""""

**Description:** Defines the aggregation function, which will take as input all the messages received by each of the
destination nodes respectively, and aggregates them together into a single representation. Note that, to define
potentially very complex function, we define this as a pipeline of aggregation operations

**Accepted values:** Array of :ref:`Aggregation operation <aggregation_operation>`.

----

Parameter: update
"""""""""""""""""

**Description:** Defines the update function. This function will be applied to each of the destination nodes, and
given the aggregated input and the current hidden state, will produce the updated hidden state.

**Accepted values:** :ref:`Update operation <update-operation>`.
