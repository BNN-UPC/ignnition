# Executing RouteNet example

## Using a small subset of data
For this example you can find the directory *data* containing a very small subset of the dataset that we used. In addition, we have included the rest of files properly filled, and thus require only to be executed.

To train the corresponding RouteNet GNN to learn to make predictions of the end-to-end metrics of a network, simply run:

```python
    python main.py
```

This command will create the GNN specified in *model_description.yaml* file. We have also included the *global_variables* file, even though right now is not used by the *model_description* file. To learn to use them, we refer the user to (https://ignnition.net/doc/global_variables/).

If you want to execute any other functionality that is not train and validate, simply change the *main.py* file, to specify the new functionality. Visit (https://ignnition.net/doc/train_and_evaluate/) for more information.

## Generate the full dataset
In the previous section we present the process to train RouteNet with the minimal dataset that we provide, even though this is clearly insufficient to obtain an accurate model. To obtain an accurate model, users may want to create the full dataset, following these steps:

### 1) Download the raw data
First of all, you must download one dataset (i.e., NSFNET (2GB), GEANT (6GB) or synth50(28.7GB), and untar them.
   
    wget "http://knowledgedefinednetworking.org/data/datasets_v0/nsfnet.tar.gz"
    wget "http://knowledgedefinednetworking.org/data/datasets_v0/geant2.tar.gz"
    wget "http://knowledgedefinednetworking.org/data/datasets_v0/synth50.tar.gz"
    tar -xvzf nsfnet.tar.gz 
    tar -xvzf geant2.tar.gz 
    tar -xvzf synth50.tar.gz

### 2) Execute the migrate file
Once you have download and untared the dataset, place yourself at the scope of the directory of the provided main.py file, and  execute:
```python
    python main.py <PATH TO DATASET> <PATH TO OUTPUT> <NUM_SAMPLES_PER_PATH>
```
To execute this file, you need to pass three arguments as parameters. First of all, the path to the untared dataset. Then the path to the directory where the new dataset will be stored. Finally, the number of samples that each of the files in the new dataset can contain. For example, passing a 100 ensures that you end up with a directory with multiple files, each of which has at most 100 samples.
It is important to consider that this process can take several minutes to finish.

### 3) Rerun
Go to the *model_description.yaml* file and ensure that the *train_dataset* path points to your new dataset.
Finally, execute:
```python
python main.py
```
    