Motivation on GNNs
==================

Graphs are an essential data type that enables the representation of
information that is relational in a properly structured manner. In this
context, GNN has recently emerged as an efficient Neural Network (NN)
technique with outstanding applications in many different fields where
data is fundamentally represented as graphs. Before *GNNs*, a common
approach to address graph-based problems was to use conventional NN
(e.g., fully-connected NN) which were fed with plain representations of
the graph that attempted to preserve its most meaningful information.
Due to the inmense impact of this representation in the performance of
the model, we typically require *ML* experts to do feature engineering
or use generic embedding techniques. Even in such scenario, however, we
still end up with a model that works as well as the relevance of the
features we handcrafted, which can be a cumbersome task in complex
scenarios.

In this context, Graph Neural Networks (GNNs) were recently proposed as
a novel NN family specifically designed to process, understand, and
model graph-structured data autonomously. Similar to the way
Convolutional NNs process images, GNNs process input graphs in raw
format, without any pre-processing. Thus, during training, GNNs learn
how to process graphs according to the problem goals (e.g., predict a
particular global metric of the graph). Moreover, GNNs are invariant to
node and edge permutations, and offer strong *relational inductive bias*
over graphs . This endows them with powerful capabilities to generalize
to other graphs of different sizes and structures not seen during the
training phase. As a result, we have witnessed a plethora of
applications where GNN has shown unprecedented generalization properties
over graphs with respect to previous NN architectures. Below we provide
several use-cases of different fields which are solved through the use
of Graph Neural Networks:

**Networking**: Modeling the performance of networks, optimizing the
routing, or scheduling jobs in data centers.

#. `List of must-read papers on GNN for communication networks <https://github.com/BNN-UPC/GNNPapersCommNets>`__
#. `RouteNet: Leveraging Graph Neural Networks for Network Modeling and Optimization in SDN <https://ieeexplore.ieee.org/abstract/document/9109574>`__
#. `Learning and Generating Distributed Routing Protocols Using Graph-Based Deep Learning <https://dl.acm.org/doi/abs/10.1145/3229607.3229610>`__

**Chemistry and biology**: Predicting complex molecular properties, drug
side effects, protein–protein interactions, or generating personalized
medication recommendations and novel compounds with some desired
properties.

#. `Neural Message Passing for Quantum Chemistry <https://arxiv.org/abs/1704.01212>`__
#. `Graph Convolutional Policy Network for Goal-Directed Molecular Graph Generation <https://arxiv.org/abs/1806.02473>`__
#. `Modeling polypharmacy side effects with graph convolutional networks <https://academic.oup.com/bioinformatics/article/34/13/i457/5045770>`__

**Physics**: Modeling interactions between particles in complex systems,
or reconstructing particle tracks in high-energy particle accelerators.

#. `Interaction Networks for Learning about Objects, Relations and Physics <https://arxiv.org/abs/1612.00222>`__
#. `Novel deep learning methods for track reconstruction <https://arxiv.org/abs/1810.06111>`__

**Mathematics**: Solving complex graph-based problems like graph
clustering, or combinatorial optimization (e.g., TSP).

#. `Learning to Solve NP-Complete Problems: A Graph Neural Network for Decision TSP <https://ojs.aaai.org/index.php/AAAI/article/view/4399>`__
#. `Weisfeiler and Leman Go Neural: Higher-Order Graph Neural Networks <https://ojs.aaai.org/index.php/AAAI/article/view/4384>`__

**Information science**: Creating recommender systems applied to
multiple products and fields

#. `Graph Neural Networks for Social Recommendation <https://arxiv.org/abs/1902.07243>`__
#. `Graph Convolutional Neural Networks for Web-Scale Recommender Systems <https://arxiv.org/abs/1806.01973>`__
#. `Inductive Matrix Completion Based on Graph Neural Networks <https://www.groundai.com/project/inductive-matrix-completion-based-on-graph-neural-networks3961/>`__
