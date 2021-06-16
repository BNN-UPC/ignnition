# Train and evaluate your model
One of the most distinctive aspects of *IGNNITION* is that the high-level definition of the architecture of the *GNN*. This implies that when designing our *GNN*, we are not concerned about many aspects that affect the training process (e.g., optimizers, losses, training duration or simply the location of the datasets to be used). Nevertheless, all this information must be specified, for which we design a specific configuration file  containing all this information. We shall now review first the following aspects.<br>
1. [Running the training and evaluation](#1-run-the-training-and-evaluation)
2. [Creation of the configuration file](#2-configuration-file)<br>


## 1. Run the training and evaluation 
### Create the model
By making the call from below, *IGNNITION* is creating a model with all the information of the model_description.yml file as well as the different training options specified. For this, the user must specify the path to the model directory where we can find the model_description.yaml file, and the training_options.yaml file. 

    model = ignnition.create_model(model_directory = './MY_EXAMPLE')

### Create the computational graph
One of the main problems that any Machine Learning Engineer has faced is to properly debug the models. Machine learning oftently perform very well, but we are rarely capable of having proper insight on its inner working. This makes it very challanging and time consuming to fix / troubleshoot a ML model.

For this, IGNNITION incorporates a debugging system which is based on producing a simplified computational graph of the model. For this, the user should make the following call over the previously created model

    model.computational_graph()

This will create a directory named "computational_graph", in the corresponding path indicated in the "train_options.yml" file. We further extend on how to visualize or interpret the output of this operation in [debugging assistant](./debugging_assistant.md). 

### Train and validation
The main goal of *IGNNITION* is precisely to be able to train a GNN model easily. For this, the user must only make the following call:

    model.train_and_validate()

This calll proceeds to train the *GNN* specified in the *model_description.yaml* file. Additionally, a process of validation will be performed throughout the training phase, so as to provide better insight on the model performance.
The behaviour during the training phase is controlled by the *train_options.yaml* file, specified in section [2. Configuration file]().

Furthermore, this operations will create (if it doesn't exist already) a directory called "CheckPoints" in the specified path. In this directory, a new directory will be created corresponding to the experiment runned (indexed by date). In it, the corresponding checkpoints will be stored, as well as the "events.out.tfevents.XXX" file, which contains the tensorboard information of the training metrics specified. Similarly as before, the user can visualize this information by running

    tensorboard --logdir <PATH TO THE "events.out.tfevents.XXX" FILE>

Then, by visiting *localhost:6006*, the user can analyse the different statistics produced during the training phase to evaluate the model (e.g., loss, mean absolute error, mean relative error, R2...)

### One-step training
Finally, we incorporated another function that is specially interesting in the context of *Reinforcement Learning*. This function allows to do training only on a batch of data passed by parameter to the function. This allows the incorporation of *IGNNITION* in real-world *Deep Reinforcement Learning (DRL)* applications that make use of the incredible power of GNNs to develop cutting-edge solutions. For this, we must only make the call shown below, passing an array of samples, each of which follows the same structure defined in [Generate your dataset](./generate_your_dataset.md).

    model.one_step_training(training_batch)

In this case, the *train_options.yaml* file needs not to indicate any training or validation dataset, as the data is directly fed by parameter.

### Evaluate
We also incorporated a very useful function which allows the user to evaluate by obtaining performance metrics of a previously trained model. More specifically, the *Evaluate* functionality takes as input the speficied validation dataset --or the array of samples passe. Then, after loading the indicated model, it will make the corresponding predictions for each of the samples, and then will compute the performance metrics of these predictions with respect to the true label found in the dataset.

For this, clearly, the user will have to indicate the path where the *trained model* can be found, as well as the metrics that we want *IGNNITION* to compute. All this information will be encoded in the *train_options.yaml* in the following fields: 
    
    validation_dataset: <PATH>

    load_model_path: <PATH>

More information regarding these fields can be found in section [2. Configuration file](). Moreover, to run this functionality, the user must only make the following call, which will return the result of the aforementioned metrics.

    model.evaluate()

### Predict
*IGNNITION* also allows to make predictions over a previously pre-trained *GNN*. For this, the user must define 2 special fields in the "train_options.yaml" file, which are:

    predict_dataset: <PATH>

    load_model_path: <PATH>

In this fields, we can specify the dataset that we aim to predict, and the location of the checkpoint of the model that we need to restore, to later be used for the predicting phase. See more details on how to fill this fields in [2. Configuration file]().

Then, *IGNNITION* will compute the corresponding prediction of each of the samples of the prediction dataset. Moreover, to run this functionality, the user must only make the following call, which will return all the computed predictions.

    model.predict()




## 2. Configuration file
In this section we review in depth the content of the *train_options.yaml* file, which will contain all the configuration parameters that ultimately define the behaviour of the specific functionality executed by the user. We must note that this file must be written in *YAML* format, which allows a very intuitve definition of all the possible fields in the form of *KEY:  VALUE*.

### Definition of the paths
At this point, we must provide the  different paths which *IGNNITION* will use to locate the information and store the results of its execution. For this, the user must fill the following fields in the train_options.ini file. Let us note that all these paths can either be absolute paths or relative paths from the directory of this file.

#### Path to the training dataset
Indicate the path pointing the training dataset, used by the *train and validation* functionality.

```yaml
train_dataset: <PATH>
```

#### Path to the validation dataset
Indicate the path pointing the validation/evaluation dataset, that will be used by the *train and validation* functionality, as well as the *evaluation* functionality.

```yaml
validation_dataset: <PATH>
```

#### Path to the prediction dataset
Defines the path to the prediction dataset, used by the *predict functionality*. Notice that this field needs only to be specified in the case that a *predict* functionality is executed. Otherwised, it will be ignored.

```yaml
predict_dataset: <PATH>
```

#### Load model path
Sometimes we might wish to use a previous checkpoint as starting point for our training process (e.g., for evaluation functionality or for predicting). For this, the user can specify the path to such checkpoints, and *IGNNITION* will use it automatically.

```yaml
load_model_path: <PATH>
```

#### Output path
Path where the *Checkpoint* and *logs* directory will be created when executing the *train and validate* functionality.

```yaml
output_path: <PATH>
```

#### Additional file path
Path to an additional *python* file that may contain implementation of specific functions, such as the implementation of a certain metric or of a certain loss function

```yaml
additional_file: <PATH>
```

#### Path to the model description file
In this case, *IGNNITION* assumes that the definition of the *GNN* --through the model description file-- is present in the very same directory as the *train_options.yaml* file itself. Hence, there is no need to speficy anything at all regarding this file.


### Model training parameters
#### Loss
Name of the loss function to use for the training of the model. It can be a name from [tf.keras.losses](https://www.tensorflow.org/api_docs/python/tf/keras/losses) library or a custom function which can be defined by the user.

```yaml
loss: [MeanSquaredError]
```

#### Optimizer
Definition of the optimizer which follows the same syntax as the [tf.keras.optimizers library](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers). Thus the user must use the exact name used in this library to reference it.

```yaml
optimizer:
  type: Adam
```

##### Additional parameters
Following the documentation of the [tf.keras.optimizers library](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers), one can for instance define different attributes that model the inner working of the desired optimizer, in this case *ADAM*. To do so, we simply include, just like the *type* attribute, any other parameter included in the afformentioned library that accepts this optimizer. Note that if no parameters are defined, *IGNNITION* will use the default values defined in the *tensorflow* library. For illustrative purpuses however, let us suppose we want to change the beta_1 value to 0.9 and the beta_2 value to 0.9 also. This can be done as follows:

```yaml
optimizer:
  type: Adam
  beta_1: 0.9
  beta_2: 0.999
```

##### Use of schedules
Finally we consider the case in which we want to define a schedule to be used with our optimizer. For this, again, we follow the syntax of of the library [tf.keras.optimizers.schedules](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules). Thus, we must only create a parameter *learning_rate*, just as we would with the tensorflow library, and pass to it the definition of the scheduler. This definition follows the same principle from before. Reference the schedule type using a valid name from the previously mentioned library, and also include any other desired parameter supported in such library (otherwise *IGNNITION* will use the default values). Below we show a simple example defining an exponential decay schedule:

```yaml
optimizer:
  type: Adam
  beta_1: 0.9
  beta_2: 0.999
  learning_rate:  # defines the schedule here
    type: ExponentialDecay
    initial_learning_rate: 0.001
    decay_steps: 80000
    decay_rate: 0.6
```
#### Metrics
Metrics defines the list of metric criterias that we want to use to evaluate our *GNN* model. This metrics will be plotted during the training and validation phase.

```yaml
metrics: [MeanAbsoluteError]
```

For this definition, the user may specify in this list any name supported by the library [tf.keras.metrics](https://www.tensorflow.org/api_docs/python/tf/keras/metrics). Additionally, the user can define a custom metric by indicating its name, and then define a function with this very same name in the *addition file*.


### Advanced options
*IGNNITION* also allows the user to costumize basically any training option. For this, in this last part of the file, the user must specify the following fields. We recommend to copy-past the provided examples, and change only the desired fields -if any-.

#### Batch size
Specify the batch_size in order to internally execute the mini-batch algorithm.

```yaml
batch_size: 32
```

#### Number of epochs
Specify the number of epochs that the algorithm must run before termination.

```yaml
epochs: 100
```

#### Epoch size
This is an optional parameter which defines the number of elements that form each of the epochs (using a natural number). Note that if this is not specified, *IGNNITION* will consider the whole dataset as a single epoch. This option is useful if the dataset is very big, as we must recall that validation is only carried out after each of the epochs.

```yaml
epoch_size: 10000
```

#### Training shuffling
True / False to indicate if the training dataset should be shuffled.

```yaml
shuffle_train_samples: True
```

#### Validation shuffling
True / False to indicate if the evaluation dataset should be shuffled.

```yaml
shuffle_validation_samples: False
```

#### Validation samples
Specify the number of evaluation samples to be used for the evaluation of our GNN.

```yaml
val_samples: 100
```

#### Validation frequency
Number of epochs after between validations.

```yaml
val_frequency: 100
```

#### K-best checkpoints
Natural number indicating the number of checkpoints that we want to keep. Note that the system will automatically keep the best $k$ checkpoints in terms of the loss.

```yaml
k_best: 5
```

#### Batch normalization
When defining a model, we can either not use any normalization at all, define a normalization function that will be applied to all the dataset, or use batch normalization. This batch normalization will apply the same normalization function for all the elements of a single batch respectively. So far, *IGNNITION* supports the use of *mean* and *max* normalization.
```yaml
batch_norm: mean
```

#### Initial epoch
When using an existing checkpoint as starting point of our *GNN*, it might be desirible to adapt also the initial epoch number. This is due to the fact that such value has implications on the learning rate (which normally gets smaller as the training advances). To do so, the user can (optionally) define the initial epoch to start the training by indicating its number (e.g., 100), which by default will take the value 0.

```yaml
initial_epoch: 100
```