# RouteNet model with support for node-level features

This implementation is based on the model described at the following paper:<br>
A. Badia-Sampera, J. Suárez-Varela, P. Almasan, K. Rusek, P. Barlet-Ros, A. Cabellos-Aparicio, "Towards more realistic network models based on Graph Neural Networks" [[Paper](https://personals.ac.upc.edu/pbarlet/papers/gnn.conext2019.pdf)], Proceedings of ACM CoNext, student workshop.

This work extends the architecture of RouteNet to support different features on forwarding devices. The paper focuses on modeling networks where devices may have variable queue size.

The datasets used in the paper are not publicly available. However, we provide the implementation of this model as a useful reference for future works.

We have also included the *global_variables* file, even though right now is not used by the *model_description* file. To learn to use them, we refer the user to (https://ignnition.org/doc/global_variables/).

