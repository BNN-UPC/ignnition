.. _what-is-a-gnn:

What is a GNN?
--------------

For the use of *IGNNITION*, we focus on one of the most basic GNN
architectures named *Message-Passing Neural Networks (MPNN)*, which is a
well-known GNN family that covers a wide range of standard GNN
architectures.

The input of a GNN is a graph *G=(V, E)* -- directed or undirected --
that comprises a set of nodes :math:`V` and some edges connecting them :math:`E`. Each node
*v* has an associated vector of a predefined size that encodes its state,
namely, the node hidden state :math:`h_v`. At the beginning of a GNN execution,
hidden state vectors :math:`h_v` are initialized with some node-related features :math:`x_v`
included in the input graph. Optionally, edges may also contain a set of
features denoted by :math:`e_{uv}` where :math:`u,v \in V`.

Once the hidden states are initialized, a message-passing algorithm is
executed according to the connections of the input graph. In this
message-passing process, three main phases can be distinguished: *(i)
Message*, *(ii) Aggregation*, and *(iii) Update*. First, every node :math:`v \in V`
sends its hidden state to all its neighbors :math:`u \in N(v)`. Then, each node applies
the *message function* :math:`M(·)` to each of the received messages respectively, to
obtain a new more refined representation of the state of its neighbors.
After this, every node merges all the computed messages from its
neighbors into a single fixed-size vector :math:`m_v` that comprises all the
information. To do this, they use a common *Aggregation function* (e.g.,
element-wise summation ). Lastly, every node applies an *Update
function* :math:`U(·)` that combines its own hidden state :math:`h_v` with the final aggregated
message from the neighbors to obtain its new hidden state. Finally, all
this message-passing process is repeated a number of iterations
:math:`T` until the node hidden states converge to some fixed values.
Hence, in each iteration, a node potentially receives --- via its direct
neighbors --- some information of the nodes that are at :math:`k` hops in the
graph. Formally, the message passing algorithm can be described as:

.. math::

    Message: \quad m_{vw}^{t+1} = M(h_v^t,h_w^t,e_{v,w}) \\[11pt]
    Aggregation: \quad m_v^{t+1} = \sum_{w \in N(v)} m_{vw}^{t+1} \\[2pt]
    Update: \quad h_v^{t+1} = U(h_v^t,m_v^{t+1})

All the process is also summarized in the figure below:

.. figure:: Images/message_passing.png
   :alt: MP

After completing the *T* message-passing iterations, a *Readout
function* :math:`R(·)` is used to produce the output of the GNN model. Particularly,
this function takes as input the final node hidden states :math:`h^t_v` and converts
them into the output labels of the model :math:`\hat{y}`:

.. math::

    Readout: \quad \hat{y} = R({h_v^T \forall v \in V})

At this point, two main alternatives are possible: *(i)* produce
per-node outputs, or *(ii)* aggregate the node hidden states and
generate global graph-level outputs. In both cases, the *Readout*
function can be applied to a particular subset of nodes.

One essential aspect of GNN is that all the functions that shape its
internal architecture is *universal*. Indeed, it uses four main
functions that are replicated multiple times along the GNN architecture:
*(i)* the *Message* , *(ii)* the *Aggregation* , *(ii)* the *Update* ,
and *(iv)* the *Readout* . Typically, at least, and are modeled by
three different Neural Networks (e.g., fully-connected NN, Recurrent NN)
which approximate those functions through a process of joint
fine-tunning (training phase) of all their internal parameters. As a
result, this dynamic assembly results in the learning of these universal
functions that capture the patterns of the set of graphs seen during
training, and which allows its generalization over unseen graphs with
potentially different sizes and structures.

Additional material
-------------------

Some users may find the previous explanation too general as, for
simplicity, we focus only on the most general scenario. For this reason,
we provide some additional material to learn about *GNNs* in more depth.

Related papers
~~~~~~~~~~~~~~

Due to the recent plethora of *GNNs*, numerous papers have been written
on this topic. We however chose two of them which we believe are very
relevant to get started on *GNNs*.

#. `The Graph Neural Network Model <https://ieeexplore.ieee.org/document/4700287>`__
#. `Graph Neural Networks: A Review of Methods and Applications <https://arxiv.org/pdf/1812.08434.pdf>`__

Blogs
~~~~~

Below we also enumerate several blogs that provide a more intuitive
overview of *GNNs*.

#. `Graph convolutional networks <https://tkipf.github.io/graph-convolutional-networks/>`__
#. `Graph Neural Network and Some of GNN Applications <https://neptune.ai/blog/graph-neural-network-and-some-of-gnn-applications>`__

Courses
~~~~~~~

Due to the fact that *GNNs* are still a very recent topic, not too many
courses have been taught covering this material. We, however, recommend
a course instructed by Standford university named `Machine learning with
graphs <http://web.stanford.edu/class/cs224w/>`__.
