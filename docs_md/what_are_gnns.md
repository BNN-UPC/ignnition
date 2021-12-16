# Backgound on GNNs
## What is a GNN?
For the use of *IGNNITION*, we focus on one of the most basic GNN architectures named *Message-Passing Neural Networks (MPNN)*, which is a well-known GNN family that covers a wide range of standard GNN architectures.

The input of a GNN is a graph *G=(V, E)* -- directed or undirected -- that comprises a set nodes <img src="https://render.githubusercontent.com/render/math?math=v \in V"> and some edges connecting them <img src="https://render.githubusercontent.com/render/math?math=e \in E"> . Each node *v* has an associated vector of predefined size that encodes its state, namely, the node hidden state <img src="https://render.githubusercontent.com/render/math?math=h_v">. At the beginning of a GNN execution, hidden state vectors <img src="https://render.githubusercontent.com/render/math?math=h_v">  are initialized with some node-related features <img src="https://render.githubusercontent.com/render/math?math=X_v"> included in the input graph. Optionally, edges may also contain a set of features denoted by <img src="https://render.githubusercontent.com/render/math?math=e_{uv}">

Once the hidden states <img src="https://render.githubusercontent.com/render/math?math=h_v"> are initialized, a message passing algorithm is executed according to the connections of the input graph. In this message-passing process, three main phases can be distinguished: *(i) Message*, *(ii) Aggregation*, and *(iii) Update*. First, every node <img src="https://render.githubusercontent.com/render/math?math=v \in V"> sends its hidden state to all its neighbors <img src="https://render.githubusercontent.com/render/math?math=U \in N(v)">. Then, each node applies the *message function* <img src="https://render.githubusercontent.com/render/math?math=M(\cdot)"> to each of the received messages respectively, to obtain a new more refained representation of the state of its neighbors. After this, every node merges all the computed messages from its neighbors into a single fixed-size vector <img src="https://render.githubusercontent.com/render/math?math=m_v"> that comprise all they information. To do this, they use a common *Aggregation function* (e.g., element-wise summation ). Lastly, every node applies an *Update function* <img src="https://render.githubusercontent.com/render/math?math=U(\cdot)"> that combines its own hidden state <img src="https://render.githubusercontent.com/render/math?math=h_v"> with the final aggregated message from the neighbors <img src="https://render.githubusercontent.com/render/math?math=m_v"> to obtain its new hidden-state. Finally, all this message-passing process is repeated a number of iterations *t=[1,T]* until the node hidden states converge to some fixed values. Hence, in each iteration <img src="https://render.githubusercontent.com/render/math?math=t=k"> a node potentially receives --- via its direct neighbors --- some information of the nodes that are at <img src="https://render.githubusercontent.com/render/math?math=k"> hops in the graph. Formally, the message passing algorithm can be described as:

\[Message: \quad m_{vw}^{t+1} = M(h_v^t,h_w^t,e_{v,w})\]
\[Aggregation: \quad m_v^{t+1} = \sum_{w \in N(v)} m_{vw}^{t+1}\]
\[Update: \quad h_v^{t+1} = U(h_v^t,m_v^{t+1})\]

All the process is also summarized in the figure below:

![MP](Images/message_passing.png)

After completing the *T* message-passing iterations, a *Readout function* <img src="https://render.githubusercontent.com/render/math?math=R(\cdots)"> is used to produce the output of the GNN model. Particularly, this function takes as input the final node hidden states <img src="https://render.githubusercontent.com/render/math?math=h_v^T"> and converts them into the output labels of the model <img src="https://render.githubusercontent.com/render/math?math=\hat{y}">:

\[Readout: \quad \hat{y} = R({h_v^T | v \in V})\]

At this point, two main alternatives are possible: *(i)* produce per-node outputs, or *(ii)* aggregate the node hidden states and generate global graph-level outputs. In both cases, the *Readout* function can be applied to a particular subset of nodes <img src="https://render.githubusercontent.com/render/math?math=V' \in G">.

One essential aspect of GNN is that all the functions that shape its internal architecture are *universal*. Indeed, it uses four main functions that are replicated multiple times along the GNN architecture: *(i)* the *Message* <img src="https://render.githubusercontent.com/render/math?math=m(\cdot)">, *(ii)* the *Aggregation* <img src="https://render.githubusercontent.com/render/math?math=aggr(\cdot)">, *(ii)* the *Update* <img src="https://render.githubusercontent.com/render/math?math=U(\cdot)">, and *(iv)* the *Readout* <img src="https://render.githubusercontent.com/render/math?math=R(\cdot)">. Typically, at least <img src="https://render.githubusercontent.com/render/math?math=M(\cdot)">, <img src="https://render.githubusercontent.com/render/math?math=U(\cdot)"> and <img src="https://render.githubusercontent.com/render/math?math=R(\cdot)"> are modeled by three different Neural Networks (e.g., fully-connected NN, Recurrent NN) which approximate those functions through a process of joint fine-tunning (training phase) of all their internal parameters'  <img src="https://render.githubusercontent.com/render/math?math=\theta">. As a result, this dynamic assembly results in the learning of these universal functions that capture the patterns of the set of graphs seen during training, and which allows its generalization over unseen graphs with potentially different sizes and structures.

## Additional material
Some user may find the previous explanation too general as, for simplicity, we focus only on the most general scenario. For this reason, we provide some additional material to learn about *GNNs* in more depth.

### Related papers
Due to the recent plethora of *GNNs*, numerous papers have been writen on this topic. We however chose two of them which we believe are very relevant to get started on *GNNs*.<br>
1. [The Graph Neural Network Model](https://ieeexplore.ieee.org/document/4700287)<br>
2. [Graph Neural Networks: A Review of Methods and Applications](https://arxiv.org/pdf/1812.08434.pdf)

### Blogs
Below we also enumerate several blogs that provide a more intuitive overview of *GNNs*.<br>
1. [Graph convolutional networks](https://tkipf.github.io/graph-convolutional-networks/)<br>
2. [Graph Neural Network and Some of GNN Applications](https://neptune.ai/blog/graph-neural-network-and-some-of-gnn-applications)

### Courses
Due to the fact that *GNNs* are still a very recent topic, not too many courses have been taught covering this material. We, however, recommend a course instructed by Standford university named [Machine learning with graphs](http://web.stanford.edu/class/cs224w/).
