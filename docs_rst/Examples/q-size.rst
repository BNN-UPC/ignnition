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

.. button::
   :text: Button Text
   :link: https://github.com/knowledgedefinednetworking/ignnition/tree/master/examples/Q-size

.. raw:: html

    <a href="<https://github.com/knowledgedefinednetworking/ignnition/tree/master/examples/Q-size" class="btn btn-neutral float-right" display="inline-block">Try Q-Size</a>



