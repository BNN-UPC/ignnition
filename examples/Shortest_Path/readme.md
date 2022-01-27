# Executing Shortest-Path example

For this example you can find the directory *data* containing the dataset that we used. In addition, we have included the rest of files properly filled, and thus require only to be executed.

To train the corresponding GNN to learn to solve the Shortest-Path routing problem, simply run:

```python
    python main.py
```

This command will create the GNN specified in *model_description.yaml* file. We have also included a *model_description_global_var.yaml* file, which contains the same model but using global variables instead. To use this other implementation, simply rename this file to be named *model_description.yaml*.

If you want to execute any other functionality that is not train and validate, simply change the *main.py* file, to specify the new functionality. Visit (https://ignnition.org/doc/train_and_evaluate/) for more information.

Finally, we recommend you to visit [our documentation](https://ignnition.org/doc/quick_tutorial/) for further details regarding this model and the complete pipeline to execute it.

