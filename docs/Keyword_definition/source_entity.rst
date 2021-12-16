.. _entity_object:

Source entity object
^^^^^^^^^^^^^^^^^^^^

This object ultimately defines how the nodes of a source entity send nodes to the destination entity. This definition
also includes the :ref:`message function <message-function-object>` which will specify how this source entity forms its
messages. To define this object, we must specify the following parameters:


.. contents::
    :local:
    :depth: 1

----

.. _entity_name:

Parameter: name
~~~~~~~~~~~~~~~

**Description:** Name of the source entity.

**Accepted values:** String. It must match the name of an entity defined previously.

.. code-block:: yaml

   name: source1

----

Parameter: message
~~~~~~~~~~~~~~~~~~

**Description:** Message function which defines how the source entity nodes form the messages to be sent to the
destination entity.

**Accepted values:** :ref:`Message function <message-function-object>`

.. _message-function-object:

Message function object:
~~~~~~~~~~~~~~~~~~~~~~~~

One of the most important aspects when defining a message passing between a source entity and a destination entity is
to specify how the source entities form their messages. To do so, and to support very complex functions, we devise a
pipeline of operations, which will be specified in :ref:`Operation object <operation-object>`. An operation performs
some calculation and then returns a reference to its output. By doing so, we can concatenate operations, by referencing
previous results to obtain increasingly more complicated results. Note that the messages will be, by default, the
result of the last operation.

Take a look at the subsection (:ref:`Operation objects <operation-object>` to find the operations accepted for this
section). We, however, introduce a new specific *Operation* which can be especially useful to define a message
function, which is the :ref:`Direct_assignment <direct-assignement>` operation.



.. _direct-assignement:

Operation: Direct_assignment
""""""""""""""""""""""""""""

This operation simply assigns the source hidden states as the message to be sent. By using it, hence, each source node will use its hidden state as the message to be sent to each of its neighbor destination node.


.. code-block:: yaml

   type: direct_assignment


Usage example:
##############

Let us put all of this together to see an example of how to define a *source_entity* in which nodes of type *entity1*
send their hidden states to the corresponding destination nodes.

.. code-block:: yaml

   source_entities:
   - name: entity1
     message:
        - type: direct_assignment

But as mentioned before, we might want to form more complicated message functions. Below we show more complicated
examples using two :ref:`Neural Network operation <neural-network-operation>`, and which illustrate the power of the
pipeline of operations. In this pipeline, we can observe that we first define a neural network that takes as input
the source entity nodes (using the keyword *source*). Then we save the input by the name a *my_output1* and we reuse
it as input of the second neural network altogether with each of the destination nodes respectively. The output of
this neural network (for each of the edges of the graph) will be the message that the source node will send to the
destination node.

.. code-block:: yaml

   source_entities:
   - name: entity1
     message:
        - type: neural_network
          input: [source]
          output_name: my_output1
        - type: neural_network
          input: [my_output1, destination]

An important note is that for the definition of neural networks in the message function, *IGNNITION* reserves the use
of *source* and *target* keywords. These keywords are used to reference the source hidden states of the entity
(in this case entity1), and to reference the destination hidden states of the target node.

.. _aggregation_operation:

Aggregation operation:
~~~~~~~~~~~~~~~~~~~~~~

This object defines the *aggregation function a*. This is to define a function that given the *k* input messages of a
given destination node *(m_1, ..., m_k)*, it produces a single aggregated message for each of the destination nodes.

.. math::

   AggregatedMessage = a(m_1, ..., m_k)

For this, we provide several keywords that reference the most common aggregated functions used in state-of-art *GNNs*,
which should be specified as follows:

.. code-block:: yaml

   aggregation:
        - type: sum/min/max/ordered/...


Below we provide more details on each of these possible aggregation functions, these being:


.. contents::
    :local:
    :depth: 1

----

Option 1: sum
"""""""""""""

This operation aggregates together all the input messages into a single message by summing the messages together.

.. math::

    AggregatedMessage_j = \sum_{i \in N(j)} m_i

Example:

.. math::

    m_1 = [1,2,3]

    m_2 = [2,3,4]

    AggregatedMessage_j  = [3,5,7]

In *IGNNITION*, this operation would be represented as:

.. code-block:: yaml

   aggregation:
       - type: sum

----

Option 2: mean
""""""""""""""

This operation aggregates together all the input messages into a single message by averaging all the messages together.

.. math::

    AggregatedMessage_j = \frac{1}{deg(j)} \sum_{i \in N(j)} m_i

Example:

.. math::

    m_1 = [1,2,3]

    m_2 = [2,3,4]

    AggregatedMessage_j = [1.5,2.5,3.5]

In *IGNNITION*, this operation would be defined as:

.. code-block:: yaml

   aggregation:
       - type: mean

----

Option 3: min
"""""""""""""

This operation aggregates together all the input messages into a single message by computing the minimum over all the
received messages.

.. code-block:: yaml

   aggregation:
       - type: min

----

Option 4: max
"""""""""""""

This operation aggregates together all the input messages into a single message by computing the maximum over all the
received messages.

.. code-block:: yaml

   aggregation:
       - type: max

----

Option 5: ordered
"""""""""""""""""

This operation produces an aggregated message which consists of an array of all the input messages. This aggregation
is intended to be used with an RNN update function. Then, the *RNN* automatically updates the hidden state by first
treating the first message, then the second message, all the way to the *k-th* message.

.. math::

    AggregatedMessage_j = (m_1|| ... ||m_k)

.. code-block:: yaml

   aggregation:
       - type: ordered

----

Option 6: attention
"""""""""""""""""""

This operation performs the attention mechanism described in the paper `Graph Attention Networks <https://arxiv.org/abs/1710.10903>`_.
Hence, given a set of input messages *(m_1, ..., m_k)*\ , it produces a set of *k* weights *(a_1, ..., a_k)*.
Then, it performs a weighted sum to end up producing a single aggregated message.


.. math::

    e_{ij} = \alpha(W * h_i, W * h_j)

    \alpha_{ij} = softmax_j(e_{ij})

    AggregatedMessage_j = \sum_{i \in N(j)} m_i * alpha_{ij}


.. code-block:: yaml

   aggregation:
       - type: attention

----

Option 7: edge-attention
""""""""""""""""""""""""

This aggregation function performs the edge-attention mechanism, described in the paper
`Edge Attention-based Multi-Relational Graph Convolutional Networks <https://www.arxiv-vanity.com/papers/1802.04944/>`_.
This is based on a variation of the previous "attention" strategy, where we follow a different approach to produce the
weights *(a_1, ..., a_k)*. We end up, similarly, producing the aggregated message through a weighted sum of the input
messages and the computed weights.

.. math::

    e_{ij} = f(m_i, m_j)

    AggregatedMessage_j = \sum_{i \in N(j)} e_{ij} * m_i

Notice that this aggregation requires a neural network *e* that will compute an attention weight for each of
the neighbors of a given destination node, respectively. Consequently, in this case, we need to include a new parameter
*nn_name* , as defined in :ref:`nn_name <param-nn-name>`. In this field, we must include the name of the NN, which
we define later on (as done for any NN). In this case, however, remember that this NN must return a single value, in
other words, the number of units of the last layer of the network must be 1. This is because we want to obtain a single
value that will represent the weight for each of the edges respectively.

.. code-block:: yaml

   aggregation:
       - type: edge_attention
         nn_name: my_network

----

Option 8: convolution
"""""""""""""""""""""

This aggregation function performs the very popular convolution mechanism, described in paper `Semi-supervised
classification with Graph Convolutional Networks <https://arxiv.org/pdf/1609.02907.pdf>`_. Again, we aim to find a
set of weights *(a_1, ..., a_k)* for the *k* input messages of a given destination node. In this case, it follows
the formulation below.

.. math::

    AggregatedMessage_j = \sum_{i \in N(j)} \frac{1}{\sqrt{deg_i * deg_j}} * h_i * W

.. code-block:: yaml

   aggregation:
       - type: convolution

----

Option 9: concat
""""""""""""""""

This aggregation function is especially thought for the cases in which we have a list of messages sent from messages of
entity type *"entity1"* and a list of messages from nodes of entity type *"entity2"*. Then, this aggregation function
will concatenate together these two lists by the axis indicated in the following field "concat_axis". Then, similarly
than with the "ordered" function, we would pass this to an *RNN*, which will update itself iteratively with all the
messages received.

Parameter: concat_axis
######################

**Description:** Axis to use for the concatenation.

**Accepted values:** 1 or 2

Given the two lists of messages:

.. math::

    M_{entity_1} = [[1,2,3],[4,5,6]]

    M_{entity_2} = [[4,5,6],[1,2,3]]


If concat_axis = 1, we will get a new message

.. math::

    AggregatedMessage_j = [[1,2,3,4,5,6], [4,5,6,1,2,3]]


If concat_axis = 2, we will get a new message

.. math::

    AggregatedMessage_j = [[1,2,3], [4,5,6],[4,5,6],[1,2,3]])


.. code-block:: yaml

   aggregation:
       - type: concat
       - concat_axis: 1

Option 10: interleave
"""""""""""""""""""""

**Description:** This aggregation concatenates both messages by interleaving them.

Given the two lists of messages:

.. math::

    M_{entity_1} = [1,2,3]

    M_{entity_2} = [4,5,6]

    AggregatedMessage_j = [1,4,2,5,3,6]


.. code-block:: yaml

   aggregation:
       - type: interleave

Option 11: neural_network
"""""""""""""""""""""""""

**Description:** So far we have looked at examples where the aggregated function is defined with a single operation
(e.g., max, min, mean...). On some occasions, however, we must build more complicated functions. This operation, thus,
allows us to take the results of previous operations and pass them through a NN to compute a new value.

**Accepted values:** :ref:`Neural network operation <neural-network-operation>`

**Example of use:**
In this case, we need to include the parameter *output_name* at the end of each of the operations that preceded the
neural network. This will store each of the results of the operations, which we will then reference in the *neural
network operation*. Let us see this with an example

.. code-block::

   aggregation:
       - type: max
         output_name: max_value
       - type: min
         output_name: min_value
       - type: attention
         output_name: attention_value
       - type: neural_network
         input: [max_value, min_value, attention_value]
         nn_name: aggregation_function

In this example, we compute the max value, the min, and the result of applying the attention to the messages received by
each of the destination nodes, respectively. Then, the neural network takes as input the results of each of the
previous operations and computes the final aggregated message, used for the update.

.. _update-operation:

Update operation:
~~~~~~~~~~~~~~~~~

In order to define the update function, we must specify a *Neural Network*. Note that the syntax will be the same no
matter if the *NN* is a *feed-forward* or an *RNN*. To define it, we must only specify two fields: which are the *type*
and the *nn_name*.

.. contents::
    :local:
    :depth: 1

Parameter: type
"""""""""""""""

**Description:** This parameter indicates the type of update function to be used
**Accepted values:** Right now the only accepted keyword is *neural_network*. We will soon however include new keywords.

.. _param-nn-name:

Parameter: nn_name
""""""""""""""""""

**Description:** Name of the Neural Network to be used for the update.

**Accepted values:** String. The name should match a *NN* created in :ref:`Step 4 <neural_networks_definition>`

Below we present an example of how an update function can be defined. Note that in this case, the update will be using
the *NN* named *my_neural_network*, and which architecture must be later defined.

.. code-block:: yaml

   update:
       type: neural_network
       nn_name: my_neural_network

