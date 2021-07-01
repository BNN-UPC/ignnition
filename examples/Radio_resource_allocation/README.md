# Radio Resource Management example

Radio resource management, such as power control -modifying the power of the transmitters in a
network-, conform a computationally challenging problem of great importance to the wireless
networking community. Due to the characteristics of these networks, that is high scalability with
low latency and high variance of their properties i.e. mobile networks, the need arises for fast and
effective algorithms to optimize resource management. Traditional algorithms such as weighted
minimum mean square error (WMMSE) as well as modern approaches which rely on convex optimization
fall short and do not scale with different networks sizes.

In this example we present an application of GNNs to solve the power control problem in wireless
networks, as presented in [[1]](#scalable-radio). We generate a synthetic dataset of
transmitter-receiver pairs which interfere with each other with some channel loss coefficients,
computed as specified in [[2]](#beamforming), and with additive Gaussian noise.

The model presented in this example follows the GNN architecture used in [[1]](#scalable-radio),
which consists of:

- Feed-forward neural network to build pair-to-pair messages using the hidden states along with
  edge information (pair to pair channel losses) and aggregating messages using element-wise
  maximum.
- Feed-forward neural network to update pairs's hidden states.
- Pass-through layer which does not modify each pair's hidden stats.

The model is trained in an self-supervised way with a custom loss function which maximizes the
weighted sum rate of the network, by using the predicted power value together with the channel
losses with other pairs and the power of the additive noise. For more details, check the paper's
discussion in [[1]](#scalable-radio).

## Running the example

For this example you can find the directory _data_ containing a very small subset of the dataset. In
particular, 1000 molecules for training and 100 for validation. In addition, we have included the
rest of framework files properly filled, and thus there are no prerequisites to execute it.

To train the corresponding Radio Resource Management GNN with the default settings, just run:

```bash
    python main.py
```

This command will create the GNN specified in [model_description](model_description.yaml) file,
with the variables specified in the [global_variables](global_variables.yaml) file. To learn more
about the implementation details, refer to the
[framework documentation](https://ignnition.net/doc/generate_your_gnn/).

If you want to execute any other functionality that is not train and validate, simply change the
[main](main.py) file, see [the documentation page](https://ignnition.net/doc/train_and_evaluate/)
for more details.

## Generate the dataset

Although minimal example data is provided, one can generate bigger datasets by executing the
[generate_dataset](generate_dataset.py) Python script. This generates synthetic wireless networks
with the model specifications to create the associated NetworkX graphs, writing them as arrays
in JSON files.

Through changing the global variables in the top of the script one can change the amount of
training/validation samples, and other properties of the generated wireless networks.

Please note that the script requires specific dependencies to generate some atomical features
which the model requires, see the script's docstring for more details.

## References

1. <a name="scalable-radio"></a>
   Y. Shen, Y. Shi, J. Zhang and K. B. Letaief,
   _Graph Neural Networks for Scalable Radio Resource Management: Architecture Design and_
   _Theoretical Analysis_, in IEEE Journal on Selected Areas in Communications, vol. 39, no. 1,
   pp. 101-115, Jan. 2021 doi: 10.1109/JSAC.2020.3036965.

2. <a name="beamforming"></a>
   Y. Shi, J. Zhang and K. B. Letaief,
   _Group Sparse Beamforming for Green Cloud-RAN_, in IEEE Transactions on Wireless Communications,
   vol. 13, no. 5, pp. 2809-2823, May 2014, doi: 10.1109/TWC.2014.040214.131770.
