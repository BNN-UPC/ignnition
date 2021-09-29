Examples
========

In order to speed up even further the process of designing a *GNN*, we
provide a number of examples implemented by *IGNNITION*. We first
include a simple example which, despite not being a real-life example,
we believe can be very helpful as a beginners guide to implement a very
simple *GNN*, this being a *GNN* solving the *Shortest path problem*.

Additionally, we provide implementations of several reference papers
that include different typologies of GNN. Our hope is that, given this
implementations, most of the custom GNN can be produced just by making
some minor changes, which should help avoid any potential designing
issue. Also, in each of the respective directories of the use-cases
presented, we also provide a file containing a single sample of the
dataset to help the user to better understand the required structure of
the dataset.

Below you can find the list of examples implemented so far by
*IGNNITION*. Stay tuned as we plan to include many more examples soon:

#. :ref:`Shortest-path <1. Shortest-path>`
#. :ref:`Graph Query Neural Networks <2. Graph Query Neural Networks>`
#. :ref:`RouteNet <3. RouteNet>`
#. :ref:`Q-size <4. Q-size>`
#. :ref:`QM9 <5. QM9>`
#. :ref:`Radio Resource Allocation <6. Radio Resource Allocation>`

1. Shortest-path
----------------

Brief description
~~~~~~~~~~~~~~~~~

The first illustrative example that we present is a *GNN* that aims to
solve the well-known *Shortest-Path* problem, which architecture is
considerably simpler than any of the other proposals that we present
below -and thus is a good starting point-. To learn more about this
model, check `quick tutorial <quick_tutorial.md>`__ where we explain in
depth the target problem, as well as the architecture of the resulting
*GNN*.

2. Graph Query Neural Networks
------------------------------

Brief description
~~~~~~~~~~~~~~~~~

This model addresses a different problem: supervised learning of
traditional routing protocols with GNN, such as shortest path or max-min
routing. To this end, the authors, in paper `Learning and generating
distributed routing protocols using graph-based deep
learning <https://www.net.in.tum.de/fileadmin/bibtex/publications/papers/geyer2018bigdama.pdf>`__,
propose a new GNN architecture that contains two entity types: routers
and interfaces, being the latter the several network interfaces of each
router in the network. Thus, this model considers a two-stage
message-passing scheme with the following structure:

**Stage1:** routers, interfaces -> interfaces

**Stage2:** interfaces -> routers

As output, the model determines the interfaces that will be used to
transmit traffic, which eventually generates the routing configuration
of the network. Another particularity of this model is in the readout
definition, where it uses a pipeline with an element-wise multiplication
and then a final prediction.

Contextualization
~~~~~~~~~~~~~~~~~

Automated network control and management has been a long standing target
of network protocols. We address in this paper the question of automated
protocol design, where distributed networked nodes have to cooperate to
achieve a common goal without a priori knowledge on which information to
exchange or the network topology. While reinforcement learning has often
been proposed for this task, we propose here to apply recent methods
from semisupervised deep neural networks which are focused on graphs.
Our main contribution is an approach for applying graph-based deep
learning on distributed routing protocols via a novel neural network
architecture named Graph-Query Neural Network. We apply our approach to
the tasks of shortest path and max-min routing. We evaluate the learned
protocols in cold-start and also in case of topology changes. Numerical
results show that our approach is able to automatically develop
efficient routing protocols for those two use-cases with accuracies
larger than 95 %. We also show that specific properties of network
protocols, such as resilience to packet loss, can be explicitly included
in the learned protocol.

MSMP Graph
~~~~~~~~~~

Below we provide a visualization of the corresponding MSMP graph. In
this representation we can observe the two different entities, this
being the interfaces (INTER) and the routers (ROUTER). Then, as
mentioned before, these exchange messages in two different stages. |MSMP GQNN definition|

Try Graph Query Neural Network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To execute this example, please download the source files at `Graph
Query Neural
Network <https://github.com/knowledgedefinednetworking/ignnition/tree/master/examples/Graph_query_networks>`__.
In this directory you will find all the necessary material regarding
*Graph Query Neural Network*, including a minimal dataset along with all
the files composing the implementation of this GNN model --model
description and training option files. Moreover, we recommend reading
the provided README file in this directory, which will guide you through
this process.

3. RouteNet
-----------

Brief description:
~~~~~~~~~~~~~~~~~~

This GNN model was proposed in paper `Unveiling the potential of Graph
Neural Networks for network modeling and optimization in
SDN <https://arxiv.org/abs/1901.08113>`__. This proposal approaches the
problem of modeling optical networks for the prediction of its
performance metrics. For this, it introduces the *link* and the *path*
entity, which are used for the message passing divided into two
different stagese:

**Stage 1:** links -> paths

**Stage 2:** paths -> links

Contextualization
~~~~~~~~~~~~~~~~~

Network modeling is a key enabler to achieve efficient network operation
in future self-driving Software-Defined Networks. However, we still lack
functional network models able to produce accurate predictions of Key
Performance Indicators (KPI) such as delay, jitter or loss at limited
cost. In this paper we propose *RouteNet*, a novel network model based
on Graph Neural Network (GNN) that is able to understand the complex
relationship between topology, routing, and input traffic to produce
accurate estimates of the per-source/destination per- packet delay
distribution and loss. RouteNet leverages the ability of GNNs to learn
and model graph-structured information and as a result, our model is
able to generalize over arbitrary topologies, routing schemes and
traffic intensity. In our evaluation, we show that RouteNet is able to
predict accurately the delay distribution (mean delay and jitter) and
loss even in topologies, routing and traffic unseen in the training
(worst case MRE=15.4%). Also, we present several use cases where we
leverage the KPI predictions of our GNN model to achieve efficient
routing optimization and network planning.

MSMP Graph
~~~~~~~~~~

Below we provide a visualization of the corresponding MSMP graph for
this use-case. In this representation we can observe the two different
entities, these being the *links* and the *paths*. Then we can observe
the message passing that they perform into two separete stages. |MSMP RouteNet definition|

Try RouteNet
~~~~~~~~~~~~

For this example, we provide the corresponding implementation of the
*model\_description.json* file and all the related files needed for the
execution in
`Routenet <https://github.com/knowledgedefinednetworking/ignnition/tree/master/examples/Routenet>`__.
In this directory you will find all the necessary material regarding
*RouteNet*, including the data and all the files composing the
implementation of this GNN model -- model description and training
options files. Moreover, we recommend reading the provided README file
in this directory, which will guide you through this process.

4. Q-size
---------

Brief Description
~~~~~~~~~~~~~~~~~

This model proposed in `Towards more realistic network models based on
Graph Neural
Networks <https://dl.acm.org/doi/10.1145/3360468.3366773>`__ aims at
estimating the src-dst performance of a network (i.e delay, jitter).
This case presents a more complex GNN architecture that contains three
entity types (links, paths, and nodes). In this case, the message
passing is divided in two stages with the following structure:

**Stage 1:** paths -> nodes, and paths -> links

**Stage 2:** nodes and links -> paths

The first stage runs two message passings separately, while in the
second stage it combines the hidden states of nodes and links and
aggregates them using a Recurrent NN.

Contextualization
~~~~~~~~~~~~~~~~~

Recently, a Graph Neural Network (GNN) model called RouteNet was
proposed as an efficient method to estimate end-to-end network
performance metrics such as delay or jitter, given the topology,
routing, and traffic of the network. Despite its success in making
accurate estimations and generalizing to unseen topologies, the model
makes some simplifying assumptions about the network, and does not
consider all the particularities of how real networks operate. In this
work we extend the architecture of RouteNet to support different
features on forwarding devices, specifically we focus on devices with
variable queue sizes, and we experimentally evaluate the accuracy of the
extended RouteNet architecture.

MSMP Graph
~~~~~~~~~~

Below we provide a visualization of the corresponding MSMP graph for
this use-case. In this representation we can observe the three different
entities, this being the links, the paths and the nodes. Then we can
observe the message passing that they perform into two separete stages.
|MSMP Q-Size definition|

Try Q-size
~~~~~~~~~~

For this example, we provide the corresponding implementation of the
*model\_description.json* file in
`Q-size <https://github.com/knowledgedefinednetworking/ignnition/tree/master/examples/Q-size>`__.
Concretely, in this directory you will find all the necessary material
regarding *Q-size*, including data as well as the model description and
training files need to train this GNN model. Moreover, we recommend
reading the provided README file in this directory, which will guide you
through this process.

5. QM9
------

Brief description
~~~~~~~~~~~~~~~~~

The `QM9
dataset <https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904>`__
contains information about 134k organic molecules containing Hydrogen
(H), Carbon (C), Nitrogen (N) and Fluorine (F). For each molecule,
computational quantum mechanical modeling was used to find each atom's
“positions” as well as a wide range of interesting and fundamental
chemical properties, such as dipole moment, isotropic polarizability,
enthalpy at 25ºC, etc.

The model presented in this example follows the GNN architecture used in
`Gilmer & Schoenholz
(2017) <https://dl.acm.org/doi/10.5555/3305381.3305512>`__, which uses a
single **atom** entity and consists of:

-  Feed-forward neural network to build *atom to atom* messages
   (single-step message passing) using the hidden states along with edge
   information (atom to atom distance and bond type).
-  Gated Recurrent Unit (GRU) to update atom's hidden states.
-  Gated feed-forward neural network as readout to compute target
   properties.

Contextualization
~~~~~~~~~~~~~~~~~

Computational chemists have developed approximations to quantum
mechanics, such as Density Functional Theory (DFT) with a variety of
functionals `Becke
(1993) <https://aip.scitation.org/doi/10.1063/1.464913>`__ and
`Hohenberg & Kohn
(1964) <https://journals.aps.org/pr/abstract/10.1103/PhysRev.136.B864>`__
to compute molecular properties. Despite being widely used, DFT is
simultaneously still too slow to be applied to large systems and
exhibits systematic as well as random errors relative to exact solutions
to Schrödinger’s equation.

Two more recent approaches by `Behler & Parrinello
(2007) <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.98.146401>`__
and `Rupp et al.
(2012) <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.108.058301>`__
attempt to approximate solutions to quantum mechanics directly without
appealing to DFT by using statistical learning models. In the first case
single-hidden-layer neural networks were used to approximate the energy
and forces for configurations of a Silicon melt with the goal of
speeding up molecular dynamics simulations. The second paper used Kernel
Ridge Regression (KRR) to infer atomization energies over a wide range
of molecules.

This approach attempts to generalize to different molecular properties
of the wider array of molecules in the QM9 dataset.

Try QM9
~~~~~~~

The files describing the model description and training parameters for
the example can be found in the `framework's
repository <https://github.com/knowledgedefinednetworking/ignnition/tree/main/examples/QM9>`__,
along with a minimal subset of the dataset for direct execution. In this
regard, we recommend reading the provided README file in this directory,
which will guide you through this process. Moreover, notice that by
default we use as target the molecules' dipole moment, but the data
provided contains all molecule properties in the original dataset to
explore other options.

6. Radio Resource Allocation
----------------------------

Brief description
~~~~~~~~~~~~~~~~~

Radio resource management, such as power control -modifying the power of
the transmitters in a network-, conform a computationally challenging
problem of great importance to the wireless networking community. Due to
the characteristics of these networks, that is high scalability with low
latency and high variance of their properties i.e. mobile networks, the
need arises for fast and effective algorithms to optimize resource
management. Traditional algorithms such as weighted minimum mean square
error (WMMSE) as well as modern approaches which rely on convex
optimization fall short and do not scale with different networks sizes.

In this example we present an application of GNNs to solve the power
control problem in wireless networks, as presented in `Shen, Y., Shi,
Y., Zhang, J., & Letaief, K. B.
(2020) <https://ieeexplore.ieee.org/abstract/document/9252917>`__. We
generate a synthetic dataset of transmitter-receiver pairs which
interfere with each other with some channel loss coefficients, computed
as specified in `Shi, Y., Zhang, J., & Letaief, K. B.
(2015) <https://ieeexplore.ieee.org/abstract/document/7120176>`__, and
with additive Gaussian noise.

The model presented in this example follows the GNN architecture used in
`Shen, Y., Shi, Y., Zhang, J., & Letaief, K. B.
(2020) <https://ieeexplore.ieee.org/abstract/document/9252917>`__, which
consists of:

-  Feed-forward neural network to build pair-to-pair messages using the
   hidden states along with edge information (pair to pair channel
   losses) and aggregating messages using element-wise maximum.
-  Feed-forward neural network to update pairs's hidden states.
-  Pass-through layer which does not modify each pair's hidden stats.

Try Radio Resource Allocation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The files describing the model description and training parameters for
the example can be found in the `framework's
repository <https://github.com/knowledgedefinednetworking/ignnition/tree/main/examples/Radio_resource_allocation>`__,
along with a minimal subset of the dataset for direct execution. In this
regard, we recommend reading the provided README file in this directory,
which will guide you through this process.

Moreover, notice that by default we use the model is trained in an
self-supervised way with a custom loss function which maximizes the
weighted sum rate of the network, by using the predicted power value
together with the channel losses with other pairs and the power of the
additive noise. For more details, check the paper's discussion in `Shen,
Y., Shi, Y., Zhang, J., & Letaief, K. B.
(2020) <https://ieeexplore.ieee.org/abstract/document/9252917>`__.

.. |MSMP GQNN definition| image:: Images/msmp_gqnn.png
.. |MSMP RouteNet definition| image:: Images/msmp_routenet.png
.. |MSMP Q-Size definition| image:: Images/msmp_q_size.png
