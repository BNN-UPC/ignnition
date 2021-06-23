Welcome to IGNNITION
====================

Graph Neural Networks (GNNs) are becoming increasingly popular in
communication networks, where many problems are formulated as graphs
with complex relationships (e.g., topology, routing, wireless channels).
However, implementing a GNN model is nowadays a complex and
time-consuming task, especially for scientists and engineers of the
networking field, which often lack a deep background on neural network
programming (e.g., TensorFlow or PyTorch). This arguably prevents
networking experts to apply this type of neural networks to their
specific problems. *IGNNITION* is a TensorFlow-based framework for fast
prototyping of GNNs. It provides a codeless programming interface, where
users can implement their own GNN models in a YAML file, without writing
a single line of TensorFlow. With this tool, network engineers are able
to create their own GNN models in a matter of few hours. *IGNNITION*
also incorporates a set of tools and functionalities that guide users
during the design and implementation process of the GNN. Check out our
`quick start tutorial <documents/quick_tutorial.md>`__ to start using IGNNITION.
Also, you can visit our `examples library <documents/examples.md>`__ with some
of the most popular GNN models applied to communication networks already
implemented.

.. figure:: documents/Images/overview_ignnition.png
   :alt: MSMP definition
   :align: center

   MSMP definition

Getting started
---------------

Installation
~~~~~~~~~~~~

Visit `installation <documents/installation.md>`__ to have a detailed tutorial
on how to install *IGNNITION* and all its necessary dependencies.

IGNNITION at a glance
~~~~~~~~~~~~~~~~~~~~~

In the section `ignnition at a glance <documents/ignnition_at_glance.md>`__, we
provide an overview of the benefits of using *IGNNITION* with respect of
traditional tools for the implementation of custom GNN models.

Quick step-by-step tutorial
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because we believe that the best way to learn is by practicing, we
provide in `quick step-by-step tutorial <documents/quick_tutorial.md>`__ an
example of how to implement a *GNN* from scratch, which should be a good
starting point for any user.

About
-----

Learn more in `About us <documents/about.md>`__, about *Barcelona Neural
Networking Center* team which has carried out the development of
*IGNNITION*.

Licence
-------

Despite being an open-source project, in section
`License <documents/license.md>`__ we provide the details on the released
license.

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   documents/ignnition_at_glance
   documents/installation
   documents/quick_tutorial

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Background On GNNs

   documents/motivation
   documents/what_are_gnns

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: User Guide

   documents/intro
   documents/model_description
   documents/generate_your_dataset
   documents/train_and_evaluate
   documents/debugging_assistant
   documents/global_variables
   documents/examples

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Misc

   documents/about
   documents/citing
   documents/contact_and_issues
   documents/mailing_list
   documents/contributing
   documents/community_bylaws
   documents/license