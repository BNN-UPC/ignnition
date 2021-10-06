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
different stages:

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
the message passing that they perform into two separete stages.

.. image::Images/msmp_routenet.png
    :align:center

.. button::
   :text: Try RouteNet
   :link: https://github.com/knowledgedefinednetworking/ignnition/tree/master/examples/Routenet

|