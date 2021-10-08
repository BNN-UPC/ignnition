# Executing RouteNet example

## Using a small subset of data
For this example you can find the directory *data* containing a very small subset of the dataset that we used. In addition, we have included the rest of files properly filled, and thus require only to be executed.

To train the corresponding RouteNet GNN to learn to make predictions of the end-to-end metrics of a network, simply run:

```python
    python main.py
```

This command will create the GNN specified in *model_description.yaml* file. We have also included the *global_variables* file, even though right now is not used by the *model_description* file. To learn to use them, we refer the user to (https://ignnition.net/doc/global_variables/).

If you want to execute any other functionality that is not train and validate, simply change the *main.py* file, to specify the new functionality. Visit (https://ignnition.net/doc/train_and_evaluate/) for more information.

## Train using the full dataset
In the previous section we presented the process to train RouteNet with the minimal dataset that we provide, even though this is clearly insufficient to obtain an accurate model. To obtain an accurate model, users may want to use the full dataset by following these steps:

### 1) Download the raw data
First of all, you must download one dataset (i.e., NSFNET (2GB), GEANT (6GB) or synth50(28.7GB), and untar them.
```python
    wget "https://knowledgedefinednetworking.org/data/datasets_v1/nsfnetbw.tar.gz"
    wget "https://knowledgedefinednetworking.org/data/datasets_v1/geant2bw.tar.gz"
    wget "https://knowledgedefinednetworking.org/data/datasets_v1/gbnbw.tar.gz"
    wget "https://knowledgedefinednetworking.org/data/datasets_v1/germany50bw.tar.gz"
    tar -xvzf nsfnetbw.tar.gz 
    tar -xvzf geant2bw.tar.gz 
    tar -xvzf gbnbw.tar.gz 
    tar -xvzf germany50bw.tar.gz
```

### 2) Migrate the dataset to the adequate format
Once you have downloaded and untared the dataset, place yourself at the scope of the directory of the provided main.py file, and  execute:
```python
    python migrate.py -d <PATH TO DATASET> -o <PATH TO OUTPUT> -n <NUM_SAMPLES_PER_FILE> -s <PERCENTAGE_TRAINING>
```
For Example:
```python
    python migrate.py -d ../../../nsfnetbw/ -o ./nsfnetMigrated/ -n 100 -s 0.8
```
To execute this file, you need to pass four arguments as parameters. First of all, the path to the untared dataset. Then the path to the directory where the new dataset will be stored. The third parameter is number of samples that each of the files in the new dataset can contain. For example, passing a 100 ensures that you end up with a directory with multiple files, each of which has at most 100 samples. Finally, the last parameter is which percentage of files are going to be used for training
It is important to consider that the migration process can take several minutes to finish.

### 3) Training
Go to the *train_options.yaml* file and ensure that the *train_dataset* path points to your new dataset.
Finally, execute:
```python
    python main.py
```

### 4) Evaluate
Once the training process finished, we can evaluate our model on a different topology than the one used during training. To do this, we need to ensure that the *predict_dataset* from *train_options.yaml* points to the desired dataset. Then, execute:
```python
    python predict.py
```
Once the model finished making the predictions over all evaluation dataset, we can plot the CDF of the relative error by executing: 
```python
    python plot_cdf.py
```