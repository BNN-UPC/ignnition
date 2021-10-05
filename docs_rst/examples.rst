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

.. include:: ./Examples/shortest_path.rst
.. include:: ./Examples/graph_query_neural_networks.rst
.. include:: ./Examples/routenet.rst


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


