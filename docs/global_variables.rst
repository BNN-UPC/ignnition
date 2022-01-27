Use of global variables
=======================

What are global variables and why are they useful?
--------------------------------------------------

Throughout the implementation of *ML* models, we constantly face the
need of making changes in the architecture of our model. For instance,
we might need to change the activation functions or the number of units
of our layers. To make these changes, we have two options. First of all,
we can simply go back to the *model_description.yaml* file, and manually
change all the entries (for instance of the activation function) for their
new value. This is however error-prone and can be a very frustrating
task.

To solve this problem, we incorporate a second possibility (optional)
which is much easier and faster than the previous, as it is based on the
use of global variables. Global variables consist of, instead of
hard-coding the specific values of a parameter of our model (e.g.,
activation function, number of units...), using a reference name for
the actual value of such parameter. With this, we manage to potentially
keep a separate file with all the specific values of the parameters
that our *GNN* uses. This has several important advantages. First of
all, this allows us to have a full picture of the most relevant
parameters of our model, which can help debug purposes.
Second of all, a single change in this file has an immediate effect on
potentially many parts of the *GNN* architecture. For instance, by
simply changing the activation function in this file (one single time),
we can change all the activation functions of our model. Finally, this
approach opens the doors to the use of automatic grid search -this being
the automatic search of the optimal values of the different parameters-,
which can have a major impact on the overall performance of our model.

How to adapt my model
---------------------

To adapt our model, we need only to make minor changes on the
*model\_description.yaml* file. Below we will show a mock example of how to adapt several parts of the architecture, and finally, we will show how
the *Shortest-path* example presented in :ref:`quick tutorial <quick-step-by-step-tutorial>`
can be adapted to use such a powerful tool.

Basic working principle
~~~~~~~~~~~~~~~~~~~~~~~

The idea consists of replacing any hard-coded value of a parameter (e.g.,
in the case of activation functions, they could be ReLU, selu...) by a
reference name of our choice. Then, we simply need to add an entry in
the *global\_variables.yaml* file with its corresponding value. Let us
see this with a very simple example of a definition of a *Neural
Network*:

.. code:: yaml

    neural_networks:
    - nn_name: readout_model
      nn_architecture:
      - type_layer: Dense
        units: my_number_of_units
        kernel_regularizer: my_regularization
        activation: my_activation
      - type_layer: Dense
        units: output_units
        kernel_regularizer: my_regularization
        activation: my_activation

In this example we can observe that the number of *units*, the
*kernel\_regularizer*, and the *activation* are all parametrized by a
reference name (instead of defining its actual value). Then, we must
simply define the following lines in the *global\_variables.yaml* file:

.. code:: yaml

    my_number_of_units: 32
    my_regularization: 0.1
    my_activation: selu

This idea can be extended to any other part of the architecture of our
*GNN* from the *model\_description.yaml* file. Note, hence, that by
changing any value of the *global\_variables.yaml* file, I would be
changing the architecture of all the parts of the *GNN* that used such
reference.

Adaptation of the Shortest-Path example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To further explain how we can use global variables, click
`here <https://github.com/knowledgedefinednetworking/ignnition/tree/main/examples/Shortest_Path/model_description_global_vars.yaml>`__.

In this file, we present the updated version of the model explained in
:ref:`quick start tutorial <quick-step-by-step-tutorial>` using the global variables
that we defined in the file
`global\_variables.yaml <https://github.com/knowledgedefinednetworking/ignnition/tree/main/examples/Shortest_Path/global_variables.yaml>`__.
