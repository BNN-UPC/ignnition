Step 3: Readout
^^^^^^^^^^^^^^^

Just as for the case of the message function, the readout function can potentially be very complex. For this, we
follow a similar approach. We define the readout as a pipeline of :ref:`Operation object <operation-object>` which
shall allow us to define very complex functions. Again, each of the operations will keep the field *output_name*
indicating the name with which we can reference/use the result of this operation in successive operations.

The main particularity for the definition of the readout is that in one of the operations (normally the last one),
will have to include the name of the *output_label* that we aim to predict. To do so, include the keyword presented
below as a property of the last *Operation* of your readout function (the output of which will be used as the output of
the *GNN*).

Another important consideration is that in this case, the user can use *entity1_initial_state* as part of the input
of an operation (where *entity1* can be replaced for any entity name of the model). With this, the operation will take
as input the initial hidden states that were initialized at the beginning of the execution, and thus, before the
message-passing phase.

Parameter: output_label
~~~~~~~~~~~~~~~~~~~~~~~

**Description:** Name referencing the labels that we want to predict, which must be defined in the dataset.

**Allowed values:** Array of strings. The names should match the labels specified in the dataset.

Let us see this with a brief example of a simple readout function based on two
:ref:`Neural Network operations <neural-network-operation>`. In this case, we apply two neural networks which are
initially to each of the nodes of type *entity1*. Then, the output is concatenated together with each of the nodes of
type *entity2* (as long that there is the same number of nodes of each entity) and then applied to the second neural
network *my_network2*. Note that the last operation includes the definition of *my_label*, which is the name of the
label found in the dataset. To specify this label, we write *$my_label* so as to indicate that this keyword refers to
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
