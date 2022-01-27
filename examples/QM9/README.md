# QM9 Quantum Chemistry example

The QM9 dataset [[1]](#qm9) contains information about 134k organic molecules containing Hydrogen (H),
Carbon (C), Nitrogen (N) and Fluorine (F). For each molecule, computational quantum mechanical
modeling was used to find each atom's “positions” as well as a wide range of interesting and
fundamental chemical properties, such as dipole moment, isotropic polarizability, enthalpy at 25ºC,
etc.

The model presented in this example follows the GNN architecture used in [[2]](#neural-mp), which
consists of:

- Feed-forward neural network to build atom to atom messages using the hidden states along with
edge information (atom to atom distance and bond type).
- Gated Recurrent Unit (GRU) to update atom's hidden states.
- Gated feed-forward neural network as readout to compute target properties.

By default we use as target the molecules' dipole moment, but the provided data contains all
molecule properties in the original dataset to explore other options.

## Running the example

For this example you can find the directory *data* containing a very small subset of the dataset. In
particular, 1000 molecules for training and 100 for validation. In addition, we have included the
rest of framework files properly filled, and thus there are no prerequisites to execute it.

To train the corresponding QM9 GNN with the default settings, just run:

```bash
    python main.py
```

This command will create the GNN specified in [model_description](model_description.yaml) file,
with the variables specified in the [global_variables](global_variables.yaml) file. To learn more
about the implementation details, refer to the
[framework documentation](https://ignnition.org/doc/generate_your_gnn/).

If you want to execute any other functionality that is not train and validate, simply change the
[main](main.py) file, see [the documentation page](https://ignnition.org/doc/train_and_evaluate/)
for more details.

## Generate the dataset

Although minimal example data is provided, one can generate bigger datasets by executing the
[generate_dataset](generate_dataset.py) Python script. This downloads the QM9 dataset and
processeses a subset of molecules to create the associated NetworkX graphs, writing them as arrays
in JSON files.

Through changing the global variables in the top of the script one can change the amount of
training/validation samples, and other properties.

Please note that the script requires specific dependencies to generate some atomical features
which the model requires, see the script's docstring for more details.

## References

1. <a name="qm9"></a>
Ramakrishnan, Raghunathan; Dral, Pavlo; Rupp, Matthias; Anatole von Lilienfeld, O. (2014):
Quantum chemistry structures and properties of 134 kilo molecules. figshare.
Collection. https://doi.org/10.6084/m9.figshare.c.978904.v5

2. <a name="neural-mp"></a>
Justin Gilmer, Samuel S. Schoenholz, Patrick F. Riley, Oriol Vinyals, and George E. Dahl. 2017.
Neural message passing for Quantum chemistry.
In *Proceedings of the 34th International Conference on Machine Learning - Volume 70*
(*ICML'17*). JMLR.org, 1263–1272.
