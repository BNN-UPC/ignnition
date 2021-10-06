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

.. button::
   :text: Try QM9
   :link: https://github.com/knowledgedefinednetworking/ignnition/tree/main/examples/QM9

|