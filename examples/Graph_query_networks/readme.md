# Executing RouteNet example

For this example you can find the directory *data* containing a very small subset of the dataset that we used, so that the user can get a general idea of what the dataset looks like --and maybe even reproduce it. Notice however that this dataset is now reach enough to obtain an accurate model.
 
In addition to this dataset, we have included the rest of files properly filled, and thus it requires only to be executed. In fact, to train the corresponding GNN model to implement the GQNN, simply run:

```python
    python main.py
```

This command will create the GNN specified in *model_description.yaml* file. We have also included the *global_variables* file, even though right now is not used by the *model_description* file. To learn to use them, we refer the user to (https://ignnition.net/doc/global_variables/).

If you want to execute any other functionality that is not train and validate, simply change the *main.py* file, to specify the new functionality. Visit (https://ignnition.net/doc/train_and_evaluate/) for more information.


    