.. _installation:

Install IGNNITION
=================

To install *IGNNTION*, we provide the user with two possibilities.
Please, ensure you use **Python 3.7 - 3.8**. Otherwise, we do not guarantee
the correct installation of dependencies.

Pip
---

The first one, which we strongly recommend, is based on the use of the
library *PyPI*. In this case, an only single command is required, which we
show below. This command will automatically install if needed all the
dependencies and then install *IGNNITION*.

.. code-block:: shell

        pip install --upgrade pip
        pip install ignnition


Source files
------------

The second possibility allows the installation from the source files
themselves. To do so, follow the steps shown below:

Download the source files
~~~~~~~~~~~~~~~~~~~~~~~~~

First of all, we must download the latest version of the code of
*Github*.

.. code-block:: shell

    wget 'https://github.com/knowledgedefinednetworking/ignnition'


Prepare the environment
~~~~~~~~~~~~~~~~~~~~~~~

Then, use the command shown underneath to install all the dependencies of
*IGNNITION*, which are listed in the *requirements.txt* file.

.. code-block:: shell

        pip install -r requirements.txt


Install IGNNITION
~~~~~~~~~~~~~~~~~

Finally, you have to install the *IGNNITION* library. For this, run the
following command:

.. code-block:: shell

        python setup.py install


Next step
---------

To continue the process of creating your first *GNN*, if you feel
confident with *GNNs*, we recommend you to check out the :ref:`User Guide <user-guide>` where you will find
all the information needed to write your new model. Check also
:ref:`examples <examples>` where you will find implementations of other
*GNNs* which might serve as a starting point for your model. In case
you don't yet feel completely confident with *GNNs*, we recommend you to
examine our :ref:`quick tutorial <quick-step-by-step-tutorial>` where we review every
step to creating a *GNN* model for a simple use-case.
