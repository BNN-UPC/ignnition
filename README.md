# IGNNITION: Fast prototyping of Graph Neural Networks for Communication Networks

*IGNNITION* is the ideal framework for users with no experience in neural network programming (e.g., TensorFlow, PyTorch). With this framework, users can design and run their own Graph Neural Networks (GNN) in a matter of a few hours.

#### Website: https://ignnition.net
#### Documentation: https://ignnition.net/doc/

*IGNNITION* is especially for you if:

You are a scientist or engineer that wants to build custom GNNs adapted to your problem (e.g., computer networks, biology, physics, chemistry, recommender systems…)

Learn more at [IGNNITION at a Glance](https://ignnition.net/doc/ignnition_at_glance/).

 ## How it works?
 <p align="center"> 
  <img src="/assets/workflow.png" width="700" alt>
</p>

Create your own GNN model in three simple steps:

1. Define a GNN architecture with an intuitive YAML interface
1. Adapt your dataset
1. Execute the training with just 3 lines of code

IGNNITION produces an optimized implementation of your GNN without writing a single line of TensorFlow.

## Quick Start
### Installation
###### Recommended: Conda environment
A [conda](https://conda.io) environment definition is available in `environment.yml`. Using it is recommended, as it
ensures that an appropriate Python version and the necessary packages are installed. To use it, first install 
[miniconda](https://docs.conda.io/en/latest/miniconda.html) and then run the following:

```
conda env create -f environment.yml --name ignnition
conda activate ignnition
```

This will create the conda environment with `ignnition` name and will activate it.  

###### Recommended: Python 3.7
Please, ensure you use Python 3.7. Otherwise, we do not guarantee the correct installation of dependencies.

You can install *IGNNITION* with the following command using *PyPI*.
```
pip install ignnition
```
Alternatively, you can install it from the source code, using the following commands. These commands first download the source code, then prepare the environment, and finally install the library:
```
wget 'https://github.com/knowledgedefinednetworking/ignnition'
pip install -r requirements.txt
python setup.py install
```
Please, find more details in our [installation guide](https://ignnition.net/doc/installation/).

### Tutorial
To get started with *IGNNITION*, we have prepared a step-by-step tutorial that explains in detail how to design a basic GNN from scratch.
Click [here to start this tutorial](https://ignnition.net/doc/quick_tutorial/).

After this tutorial, you should be prepared to:
- Start designing [your own GNN model from scratch](https://ignnition.net/doc/intro/).
- Reuse any model from our [examples library](https://ignnition.net/doc/examples/) and adapt it to your needs.

Please, follow the [documentation](https://ignnition.net/doc/) to know all the details of this framework.

## Main Contributors
#### D. Pujol-Perich, J. Suárez-Varela, Miquel Ferriol, A. Cabellos-Aparicio, P. Barlet-Ros.

[Barcelona Neural Networking center](https://bnn.upc.edu/), Universitat Politècnica de Catalunya

This software is part of a project that has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 871528.


 <p align="center"> 
  <img src="/assets/ngi_european_flag.png" width="400" alt>
</p>

## License
See [LICENSE](LICENSE) for full of the license text.


```
Copyright Copyright 2020 Universitat Politècnica de Catalunya

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
