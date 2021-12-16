2. Graph Query Neural Networks
------------------------------

Brief description
~~~~~~~~~~~~~~~~~

This model addresses a different problem: supervised learning of
traditional routing protocols with GNN, such as shortest path or max-min
routing. To this end, the authors, in the paper `Learning and generating
distributed routing protocols using graph-based deep
learning <https://www.net.in.tum.de/fileadmin/bibtex/publications/papers/geyer2018bigdama.pdf>`__,
propose a new GNN architecture that contains two entity types: routers
and interfaces, being the latter the several network interfaces of each router in the network. Thus, this model considers a two-stage
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

Automated network control and management have been a long-standing target
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
mentioned before, these exchange messages in two different stages.

.. image::Images/msmp_gqnn.png
    :align:center

Try Graph Query Neural Network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To execute this example, please download the source files at `Graph
Query Neural
Network <https://github.com/knowledgedefinednetworking/ignnition/tree/main/examples/Graph_query_networks>`__.
In this directory, you will find all the necessary material regarding
*Graph Query Neural Network*, including a minimal dataset along with all
the files composing the implementation of this GNN model --model
description and training option files. Moreover, we recommend reading
the provided README file in this directory, which will guide you through
this process.

.. button::
   :text: Try Graph Query Neural Networks
   :link: https://github.com/knowledgedefinednetworking/ignnition/tree/main/examples/Graph_query_networks

|
