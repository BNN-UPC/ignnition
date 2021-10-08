.. _operation-object:

Operation objects
-----------------

We now review the different options of *Operations* that *IGNNITION* allows, and which can be used in many of the parts
of the *GNN* (e.g., message function, update function, readout function...). All these possible operations are:


.. contents::
    :local:
    :depth: 1

Operation 1: product
~~~~~~~~~~~~~~~~~~~~~~

This operation will perform the product of two different inputs. Let us go through the different parameters that we
can tune to customize this operation.

.. contents::
    :local:
    :depth: 1

----

Parameter: input
""""""""""""""""

**Description:** Defines the set of inputs to be fed to this operation.

**Allowed values:** Array of two strings, defining the two inputs of the *product operation*.

Notice that if a string from the input references a feature from the dataset, the name must always be preceeded by a
# symbol. This will indicate *IGNNITION* that such keyword references a value present in the dataset.

----

Parameter: output_name
""""""""""""""""""""""

**Description:** Defines the name by which we can reference the output of this operation if successive operations.

**Allowed values:** String

----

Parameter: type_product
"""""""""""""""""""""""

**Description:** Defines the type of product that we use (e.g., element-wise, matrix multiplication, dot-product)

**Allowed values:** [dot_product, element_wise, matrix_mult]

Let us explain in more detail what each of the following keywords stands for:


.. contents::
    :local:
    :depth: 1

----

Option 1: dot_product
#####################

**Description:** Computes the dot product between two inputs *a* and *b*. Note that if the inputs are two arrays
*a = (a_1, a_2, ... , a_k)* and *b = (b_1, ,b_2, ... , b_k)*, then the dot product is applied to *a_i* and *b_i*
respectively.

**Allowed values:** String. Name of an entity or output of a previous operation.

Below we show an example of a readout function which first computes the *dot_product* between nodes of type *entity1*
and *entity2*\ , respectively. Then, the result of each of these operations are passed to a *Neural Network* that
compute the prediction.

.. code-block:: yaml

   readout:
   - type: product
     type_product: dot_product
     input: [entity1, entity2]
     nn_name: my_network1
     output_label: output1
   - type: neural_network
     input: [output1, entity2]
     nn_name: my_network2
     output_label: [$my_label]

----

Option 2: element_wise
######################

**Description:** Computes the element-wise multiplication between two inputs *a* and *b*. Note that if the inputs are
two arrays *a = (a_1, a_2, ... , a_k)* and *b = (b_1, ,b_2, ... , b_k)*\ , then the element-wise multiplication is
applied to *a_i* and *b_i* respectively.

**Allowed values:** String. Name of an entity or output of a previous operation.

Below we show an example of a readout function which first computes the *element_wise* multiplication between nodes of
type *entity1* and *entity2*, respectively. Then, the result of each of these operations are passed to a *Neural
Network* that compute the prediction.

.. code-block:: yaml

   readout:
   - type: product
     type_product: dot_product
     input: [entity1, entity2]
     nn_name: my_network1
     output_label: output1
   - type: neural_network
     input: [output1, entity2]
     nn_name: my_network2
     output_label: [$my_label]

----

Option 3: matrix_mult
#####################

**Description:** Computes the matrix multiplication between two inputs *a* and *b*. Note that if the inputs are two
arrays *a = (a_1, a_2, ... , a_k)* and *b = (b_1, ,b_2, ... , b_k)*\ , then the matrix multiplication is applied to
*a_i* and *b_i* respectively.

**Allowed values:** String. Name of an entity or output of a previous operation.

Below we show an example of a readout function which first computes the *dot_product* between nodes of type *entity1*
and *entity2*\ , respectively. Then, the result of each of these operations are passed to a *Neural Network* that
compute the prediction.

----

.. _neural-network-operation:

Operation 2: neural_network
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Similarly to the neural_network operations used in the *message* or *update* function, we just need to reference the
neural network to be used, and provide a name for the output. Then, given some input (:math:`a`) and a neural network that we
define (:math:`f`), this operation performs the following:

.. math::

    output\_name = f(a)


Below we show a code-snipped of what a *neural_network* operation would look like, and we present afterward each of
its possible options. This neural network takes as input all the states of the nodes of type *entity1* , and pass
them (separately) to our *NN* named *my_network*. Finally it stores the results of these operations in *my_output*.

.. code-block:: yaml

   - type: neural_network
     input: [entity1]
     nn_name: my_network
     output_name: my_output

We can now review in more depth each of its available parameters:


.. contents::
    :local:
    :depth: 1

----

Parameter: input
""""""""""""""""

**Description:** Defines the set of inputs to be fed to this operation.
**Allowed values:** Array of strings. If this neural network is part of the readout, you can use *entity1_initial_state*
to reference the initial values of the hidden-states of *entity1*. Note that *entity1* can be replaced for any entity
name of the model.

An important consideration is that all the strings in the input that reference a features --that is present in the
dataset-- must be proceeded by a # symbol. This will indicate *IGNNITION* that such keyword references a value from
the dataset.

----

Parameter: nn_name
""""""""""""""""""

**Description:** Name of the neural network (:math:`f`), which shall then used to define its actual
architecture in :ref:`Step 4 <neural_networks_definition>`.

**Allowed values:** String. This name should match the one from one of the neural networks defined.

----

Parameter: output_name
""""""""""""""""""""""

**Description:** Defines the name by which we can reference the output of this operation, to be then used in
successive operations.

**Allowed values:** String

An example of the use of this operation is the following *message* function (based on a pipe-line of two different
operations):

.. code-block:: yaml

   message:
   - type: neural_network
     input: [entity1]
     nn_name: my_network1
     output_name: my_output

   - type: neural_network
     input: [my_output]
     nn_name: my_network2

With this, hence, we apply two successive neural networks, which is just a prove of some of the powerful
operations that we can define.

----

Operation 3: pooling
~~~~~~~~~~~~~~~~~~~~

The use of this operation is key to make global predictions (over the whole graph) instead of node predictions. This
allows to take a set of input (:math:`a_1, ... , a_k`) and a defined function (:math:`g`), to obtain a single resulting
output. This is:

.. math::

    output\_name = g(a_1, ..., a_k)

For this, we must define, as usual, the *output_name* field, where we specify the name for the output of this operation.
Additionally, we must specify which function (g) we want to use. Let us see how this operation would look like if used
to define a *readout* function to make global predictions over a graph. In this example we again define a pipe-line of
operations, first of all by pooling all the nodes of type *entity1* together into a single representation (which is
stored in my_output. Then we define a neural network operation which takes as input this pooled representation and
applies it to a *NN* which aims to predict our label *my_label*.

.. code-block:: yaml

   readout:
   - type: pooling
     type_pooling: sum/mean/max
     input: [entity1]
     output_name: my_output

   - type: neural_network
     input: [my_output]
     nn_name: readout_model
     output_label: [$my_label]

Again, we now present the new keyword that is characteristic from this specific operation:

Parameter: type_pooling:
""""""""""""""""""""""""

**Description:** This field defines the pooling operation that we want to use to reduce a set of inputs
(a_1, ... , a_k) to a single resulting output.

**Allowed values:** Below we define the several values that this field *type_pooling* can take:

Let us now explain in depth what each of the possible types of pooling that *IGNNITION* currently supports:

.. contents::
    :local:
    :depth: 1

----

Option 1: sum
#############

This operations takes the whole set of inputs :math:`(a_1, ... , a_k)`, and sums them all together.

.. math::

    output\_name = \sum_{n=1}^{n=k} a_n

.. code-block:: yaml

   - type: pooling
     type_pooling: sum
     input: [entity1]

----

Option 2: max
#############

This operations takes the whole set of inputs :math:`(a_1, ... , a_k)`, and outputs the its max.

.. math::

    output\_name = \max(a_1, ... , a_k))

.. code-block:: yaml

   - type: pooling
     type_pooling: max
     input: [entity1]

----

Option 3: mean
##############

This operations takes the whole set of inputs :math:`(a_1, ... , a_k)`, and calculates their average.

.. math::

    output\_name = \frac{1}{k} \sum_{n=1}^{n=k} a_n

.. code-block:: yaml

   - type: pooling
     type_pooling: mean
     input: [entity1]
