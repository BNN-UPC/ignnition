# Computing the Shortest Path in graphs with GNNs (Quick-start tutorial)

This is a basic example of a Message-Passing Neural Network applied to compute the shortest path in graphs. Please, follow our quick-start tutorial at this link for further details:
https://ignnition.org/doc/quick_tutorial.html

For this example you can find a dataset in the directory *data*. In addition, we have included the rest of files properly filled, and all is ready for direct execution.

To train the GNN model you can simply run:

```python
    python main.py
```

This command build the GNN model described in *model_description.yaml* file. We have also included a *model_description_global_var.yaml* file, which contains the same model but using global variables instead. To use this other implementation, simply rename this file to be named *model_description.yaml*.

You can change the *main.py* file to execute any other functionality apart from train and validate. Please, visit the documentation for more details (https://ignnition.org/doc).



