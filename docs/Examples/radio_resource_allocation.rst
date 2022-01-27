6. Radio Resource Allocation
----------------------------

Brief description
~~~~~~~~~~~~~~~~~

Radio resource management, such as power control -modifying the power of
the transmitters in a network-, conforms to a computationally challenging
problem of great importance to the wireless networking community. Due to
the characteristics of these networks, that is high scalability with low
latency and high variance of their properties i.e. mobile networks, the need arises for fast and effective algorithms to optimize resource
management. Traditional algorithms such as weighted minimum mean square
error (WMMSE) as well as modern approaches which rely on convex
optimization fall short and do not scale with different networks sizes.

In this example, we present an application of GNNs to solve the power
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
-  Feed-forward neural network to update pairs' hidden states.
-  Pass-through layer which does not modify each pair's hidden stats.

.. button::
   :text: Try Radio Resource Allocation
   :link: https://github.com/knowledgedefinednetworking/ignnition/tree/main/examples/Radio_resource_allocation

|
