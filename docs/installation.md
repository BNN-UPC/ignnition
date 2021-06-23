# Install IGNNITION
To install *IGNNTION*, we provide the user with two possibilities. Note that both of them require to be working with a ***Python* version 3.5 or greater**.

## Pip
The first one, which we strongly recommend, is based on the use of the library *PyPI*. In this case, only single command is required, which we show below. This command will automatically install if needed all the dependencies, and then install *IGNNITION*.

```
    pip install ignnition
```

## Source files
The second possibility allows the installation from the source files themselves. To do so, follow the steps shown below:

### Download the source files
First of all we must download the latest version of the code of *Github*.

```
wget 'https://github.com/knowledgedefinednetworking/ignnition'
```

### Prepare the enviornment
Then, use command shown underneath to install all the dependencies of *IGNNITION*, which are listed in the *requirements.txt* file.

```
    pip install -r requirements.txt
```

### Install IGNNITION
Finally, you have to install the *IGNNITION* library. For this, run the following command:

```
    python setup.py install
```

## Next step
To continue the process of creating your first *GNN*, if you feel confident with *GNNs*, we recommend you to check out the [user guide](model_description.md#generate-your-gnn) where you will find all the information needed to write your new model. Check also [examples](examples.md) where you will find implementations of other *GNNs* which might serve as starting point for your own model. In case you don't yet feel complitely confident with *GNNs*, we recommend you to examine our [quick tutorial](quick_tutorial.md) where we review every step to create a *GNN* model for a simple use-case.
