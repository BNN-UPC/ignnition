# Keyword definition
In this section we will focus in more depth on what are the keywords available to design each of the sections that themselves define the GNN, and how to use them. More specifically, we will cover the keywords for each of the following sections. 

- [Step 1: Entity definition](#step-1-entity-definition)<br>
- [Step 2: Message-passing phase](#step-2-message-passing-phase)<br>
- [Step 3: Readout](#step-3-readout)<br>
- [Step 4: Internal Neural Network definition](#step-4-internal-neural-networks)


## Step 1: Entity definition
In order to create the entities, we must define a list "entities". For this, we must define an object "Entity". We shall now describe the different keywords that the user must / can define to model the new entity, these being:<br>
- [Parameter: name](#parameter-name)<br>
- [Parameter: state_dim](#parameter-state_dim)<br>
- [Parameter: initial_state](#parameter-initial_state)

---
### Parameter: name
**Description:** Name that we assing to the new entity. This name is important as we will use it from now on to reference the nodes that belong to this entity.

**Accepted values:** String of the choice of the user.

E.g., below we show how we would define an entity of name *entity1*.

```yaml
name: entity1
```
---
### Parameter: state_dim
**Description:** Dimension of the hidden states of the nodes of this entity.

**Accepted values:** Natural number

```yaml
state_dim: 32
```
---
### Parameter: initial_state
**Description:** Array of Operation object defining incrementally the initial_state.

**Accepted values:** Array of [Operation objects](#operation-object).


## Step 2: Message-passing phase
Now we define the keywords that the user can use to design the message passing phase of the present *GNN*. To do so, we will cover the following keywords:<br>
- [Parameter: num_iterations](#parameter-num_iterations)<br>
- [Parameter: stages](#parameter-stages)

### Parameter: num_iterations
**Description:** Number of times that all the stages must be repeated (iteration of the message passing phase).

**Accepted values:** Natural number (Normally between 3 and 8)

```yaml
num_iterations: 8
```
---
### Parameter: stages
**Description:** Stages is the array of stage object which ultimately define all the parts of the message passing.

**Accepted values:** Array of [Stage objects](#stage, each of which represents a time-step of the algorithm.


## Stage:
To define a stage, we must define all the single message passings that take place during that stage (a given time-step of the algorithm). This is to define all the single message-passing which define how potentially many entities send messages to a destination entity.

### Parameter: stage_message_passings
**Description:** Contains the single message-passings (the process of one entity nodes sending messages to another one), which we assign to this stage (time-step of the algorithm)

**Accepted values:** Array of [Single message-passing objects](#single-message-passing).


## Single message-passing:
This object defines how the nodes of potentially many entity types send messages simultaniously to the nodes of a given destination entity. To do so, we must define the following parameters:<br>

- [Parameter: destination](#parameter-destination)<br>
- [Parameter: source_entities](#parameter-source_entities)<br>
- [Parameter: aggregation](#parameter-aggregation)<br>
- [Parameter: update](#parameter-update)


### Parameter: destination entity
**Description:** Name of the destination entity of this single message-passing. In other words, the entity nodes receiving the messages.

**Accepted values:** String. It must match the name of an entity previously defined (see [entity name](#parameter:-name)).

```yaml
destination_entity: my_dst_entity
```
---
### Parameter: source_entities
**Description:** Array of the source entities sending messages to the destination entity (defined before) in this single message-passing. This is, all these sending entities will send messages simultaniously to the defined destination entity.

**Accepted values:** Array of [Souce entity objects](#source-entity).

---
### Parameter: aggregation
**Description:** Defines the aggregation function, which will take as input all the messages received by each of the destination nodes respectively, and aggregates them together into a single representation. Note that, to define potentially very complex function, we define this as a pipeline of aggregation operations
**Accepted values:** Array of [Aggregation operation](#aggregation-operation).

---
### Parameter: update
**Description:** Defines the update function. This function will be applied to each of the destination nodes, and given the aggregated input and the current hidden state, will produce the updated hidden-state.

**Accepted values:** [Update operation](#update-operation).


## Source entity object:
This object ultimately defines how the nodes of a source entity send nodes to the destination entity. This definition also includes the [message function](#message-function-object) which will specify how this souce entity forms its messages. To define this object, we must specify the following parameters:

- [Parameter: name](#parameter-name)<br>
- [Parameter: message](#parameter-message)

---
### Parameter: name
**Description:** Name of the source entity.

**Accepted values:** String. It must match the name of an entity defined previously.

```yaml
name: source1
```
---
### Parameter: message
**Description:** Message function which defines how the source entity nodes form the messages to be sent to the destination entity.

**Accepted values:** [Message function](#message-function-object)


### Message function object:
One of the most important aspects when defining a message passing between a source entity and a destination entity is to specify how the source entities form their messages. To do so, and to support very complex functions, we device a pipe-line of operations, which will be specified in [Operation object](#opeartion-object). An operation performs some calculation and then returns a reference to its output. By doing so, we can concatenate operations, by referencing previous results to obtain increasingle more complicated results. Note that the messages will be, by default, the result of the last operation.

Take a look at the subsection ([Operation objects](#operation-object) to find the operations accepted for this sections). We, however, introduce a new specific *Operation* which can be specially usefull to define a message function, which is the [Direct_assignment](#operation:-direct_assignment) operation.

#### Operation: Direct_assignment
This operation simply assigns the source hidden states as the message to be sent. By using it, hence, each source node will use its hidden state as the message to be send to each of its neighbour destination node.

```yaml
type: direct_assignment
```

#### Usage example:
Let us put all of this together to see an example of how to define a *source_entity* in which nodes of type *entity1* send their hidden states to the corresponding destination nodes.

```yaml
source_entities:
- name: entity1
  message:
     - type: direct_assignment
```

But as mentioned before, we might want to form more complicated message functions. Below we show a more complicated examples using two [Neural Network operation](#neural-network-operation), and which illustrate the power of the pipe-line of operations. In this pipe-line, we can observe that we first define a neural network which take as input the source entity nodes (using the keyword *source*). Then we save the input by the name a *my_output1* and we reuse it as input of the second neural network altogether with each of the destination nodes respectively. The output of this neural network (for each of the edges of the graph) will be the message that the source node will send to the destination node.


```yaml
source_entities:
- name: entity1
  message:
     - type: neural_network
       input: [source]
       output_name: my_output1
     - type: neural_network
       input: [my_output1, target]
```

An important note is that for the definition of neural networks in the message function, *IGNNITION* reserves the use of *source* and *target* keywords. These keywords are used to reference to the source hidden states of the entity (in this case entity1), and to reference the destination hidden states of the target node.

### Aggregation operation:
This object defines the *aggregation function a*. This is to define a function that given the *k* input messages of a given destination node *(m_1, ..., m_k)*, it produces a single aggreagated message for each of the destination nodes.

```yaml
aggregated_message = a(m_1, ..., m_k)
```

For this, we provide several keywords that reference the most common aggregated functions used in state-of-art *GNNs*, which should be specified as follows:

```yaml
aggregation:
     - type: sum/min/max/ordered/...
```
Below we provide more details on each of this possible aggregation functions, these being:<br>

- [Option 1: sum](#option-1-sum)<br>
- [Option 2: mean](#option-2-sum)<br>
- [Option 3: min](#option-3-sum)<br>
- [Option 4: max](#option-4-sum)<br>
- [Option 5: ordered](#option-5-sum)<br>
- [Option 6: attention](#option-6-sum)<br>
- [Option 7: edge_attention](#option-7-sum)<br>
- [Option 8: convolution](#option-8-sum)<br>
- [Option 9: concat](#option-9-sum)<br>
- [Option 10: interleave](#option-10-sum)

---
#### Option 1: sum
This operation aggregates together all the input messages into a single message by summing the messages together.

\(aggregated\_message_j = \sum_{i \in N(j)} m_i\)

Example: 

\(m_1 = [1,2,3]\)

\(m_2 = [2,3,4]\)
         
\(aggregated\_message_j = [3,5,7]\)

In *IGNNITION*, this operation would be represented as:

```yaml
aggregation:
    - type: sum
```

---
#### Option 2: mean
This operation aggregates together all the input messages into a single message by averaging all the messages together.

\(aggregated\_message_j = \frac{1}{deg(j)} \sum_{i \in N(j)} m_i\)

Example: m_1 = [1,2,3]
         m_2 = [2,3,4]
         aggregated_message_j = [1.5,2.5,3.5]

In *IGNNITION*, this operation would be defined as:

```yaml
aggregation:
    - type: mean
```

---
#### Option 3: min
This operation aggregates together all the input messages into a single message by computing the minimum over all the received messages.

```yaml
aggregation:
    - type: min
```

---
#### Option 4: max
This operation aggregates together all the input messages into a single message by computing the maximum over all the received messages.

```yaml
aggregation:
    - type: max
```

---
#### Option 5: ordered
This operation produces an aggregated message which consists of an array of all the input messages. This aggregation is intended to be used with a RNN udpate function. Then, the *RNN* automatically updates the hidden state by first treating the first message, then the second message, all the way to the *k-th* message.

\(aggregated\_message_j = (m_1|| ... ||m_k)\)

```yaml
aggregation:
    - type: ordered
```
---
#### Option 6: attention
This operation performs the attention mechanism described in paper [Graph Attention Networks](https://arxiv.org/abs/1710.10903). Hence, given a set of input messages *(m_1, ..., m_k)*, it produces a set of *k* weights *(a_1, ..., a_k)*. Then, it performs a weighted sum to end up producing a single aggregated message.

\(e_{ij} = \alpha(W * h_i, W * h_j)\)

\(\alpha_{ij} = softmax_j(e_{ij})\)

\(aggregated\_message_j = \sum_{i \in N(j)} m_i * alpha_{ij}\)

```yaml
aggregation:
    - type: attention
```
---
#### Option 7: edge-attention
This aggregation function performs the edge-attention mechanism, described in paper [Edge Attention-based Multi-Relational Graph Convolutional Networks](https://www.arxiv-vanity.com/papers/1802.04944/). This is based on a variation of the previous "attention" strategy, where we follow a different approach to produce the weights *(a_1, ..., a_k)*. We end up, similarly, producing the aggregated message through a weighted sum of the input messages and the computed weights.

\(e_{ij} = f(m_i, m_j)\)

\(aggregated\_message_j = \sum_{i \in N(j)} e_{ij} * m_i \)

Notice that this aggregation requires of a neural network *e* that will compute an attention weight for each of the neighbors of a given destination node, respectively. Consequently, in this case, we need to include a new parameter *nn_name*, as defined in [nn_name](####parameter-nn_name). In this field, we must include the name of the NN, which we define later on (as done for any NN). In this case, however, remember that this NN must return a single value, in other words, the number of units of the last layer of the network must be 1. This is because we want to obtain a single value that will represent the weight for each of the edges respectively.

```yaml
aggregation:
    - type: edge_attention
      nn_name: my_network
      
```
---
#### Option 8: convolution
This aggregation function performs the very popular convolution mechanism, described in paper [Semi-supervised classification with Graph Convolutional Networks](https://arxiv.org/pdf/1609.02907.pdf). Again, we aim to find a set of weights *(a_1, ..., a_k)* for the *k* input messages of a given destination node. In this case, it follows the formulation below.

\(aggregated\_message_j = \sum_{i \in N(j)} \frac{1}{\sqrt(deg_i * deg_j)} * h_i * W \)

```yaml
aggregation:
    - type: convolution
```
---
#### Option 9: concat
This aggregation function is specially thought for the cases in which we have a list of messages sent from messages of entity type *"entity1"* and a list of messages from nodes of entity type *"entity2"*. Then, this aggregation function will concatenate together this two lists by the axis indicated in the following field "concat_axis". Then, similarly than with the "ordered" function, we would pass this to an *RNN*, which will update itself iteratively with all the messages received.

##### Parameter: concat_axis
**Description:** Axis to use for the concatenation. 

**Accepted values:** 1 or 2

Given the two lists of messages from "entity1" \([[1,2,3],[4,5,6]]\) and from "entity2" \([[4,5,6],[1,2,3]]\).

If concat_axis = 1, we will get a new message 

\( aggregated\_message_j = [[1,2,3,4,5,6], [4,5,6,1,2,3]]\)

If concat_axis = 2, we weill get a new message 

\(aggregated\_message_j = [[1,2,3], [4,5,6],[4,5,6],[1,2,3]]\)

---
#### Option 10: interleave
**Description:** Axis to use for the concatenation. 

**Accepted values:** 1 or 2

#### Option 11: neural_network
**Description:** So far we have looked at examples where the aggregated function is defined with a single operation (e.g., max,min,mean...). In some ocasions, however, we must build more complicated functions. This operation, thus, allows to take the results of previous operations and pass them through a NN to compute a new value.
**Accepted values:** [Neural network operation](#operation-2-neural_network)

**Example of use:**<br>
In this case, we need to include the parameter *output_name* at the end of each of the operations that preceed the neural network. This will store each of the results of the operations, which we will then reference in the *neural network operation*. Let us see this with an example

```
aggregation:
    - type: max
      output_name: max_value
    - type: min
      output_name: min_value
    - type: attention
      output_name: attention_value
    - type: neural_network
      input: [max_value, min_value, attention_value]
      nn_name: aggregation_function
```
In this example we compute the max value, the min and the result of applying the attention to the messages received by each of the destination nodes, respectively. Then, the neural network takes as input the results of each of the previous operations, and computes the final aggregated message, used for the update.

### Update operation:
In order to define the update function, we must specify a *Neural Network*. Note that the syntax will be the same no matter if the *NN* is a *feed-forward* or a *RNN*. To define it, we must only specify two fields: which are the *type* and the *nn_name*.<br>

- [Parameter: type](#parameter-type)<br>
- [Parameter: nn_name](#parameter-nn_name)

#### Parameter: type
**Description:** This parameter indicates the type of update function to be used
**Accepted values:** Right now the only accepted keyword is *neural_network*. We will soon however include new keywords.
---
#### Parameter: nn_name
**Description:** Name of the Neural Network to be used for the upate.
**Accepted values:** String. The name should match a *NN* created in [Step 4](#step-4-neural-network-architectures)

Below we present an example of how an update function can be defined. Note that in this case the update will be using the *NN* named *my_neural_network*, and which architecture must be later defined.

```yaml
update: 
    type: neural_network
    nn_name: my_neural_network
```

## Step 3: Readout
Just as for the case of the message function, the readout function can potentially be very complex. For this, we follow a similar approach. We define the readout as a pipe-line of [Operation object](#operation-object) which shall allow us to define very complex functions. Again, each of the operations will keep the field *output_name* indicating the name with which we can reference/use the result of this operation in successive opeartions.

The main particularity for the defintion of the readout is that in one of the operations (normally the last one), will have to include the name of the *output_label* that we aim to predict. To do so, include the keyword presented below as a property of last *Operation* of your readout function (the output of which will be used as output of the *GNN*).

Another important consideration is that in this case, the user can use *entity1_initial_state* as part of the input of an operation (where *entity1* can be replaced for any entity name of the model). With this, the operation will take as input the initial hidden states that were initialized at the beginning of the execution, and thus, before the message-passing phase.


### Parameter: output_label
**Description:** Name referencing the labels that we want to predict, which must be defined in the dataset.

**Allowed values:** Array of strings. The names should match the labels specified in the dataset.

Let us see this with a brief example of a simple readout function based on two [Neural Network operations](#neural-network-operation). In this case we apply two neural networks which are intially to each of the nodes of type *entity1*. Then, the output is concatenated together with each of the nodes of type *entity2* (as long that there is the same number of nodes of each entity) and then applied to the second neural network *my_network2*. Note that the last operation includes the definition of *my_label*, which is the name of the label found in the dataset. To specify this label, we write *$my_label* so as to indicate that this keywords refers to data that *IGNNITION* can find in the corresponding dataset.

```yaml
readout:
- type: neural_network
  input: [entity1]
  nn_name: my_network1
  output_label: output1
- type: neural_network
  input: [output1, entity2]
  nn_name: my_network2
  output_label: [$my_label]
```

Notice, however, that *output_label* may contain more than one label. For instance, consider the case in which we want that the readout function predicts two properties of a node, namely *label1* and *label2*. For simplicity, let us considert these labels to be single values --even though the same proceduce applies when they represent 1-d arrays. For this, we make the following adaptations of the previous model: 

```yaml
readout:
- type: neural_network
  input: [entity1]
  nn_name: my_network1
  output_label: output1
- type: neural_network
  input: [output1, entity2]
  nn_name: my_network2
  output_label: [$label1, $label2]
```

In this case, hence, *my_network2* will output two predictions, one for each of the target labels. Then, *IGNNITION* will internally process this and backpropagate accordingly, so as to force the GNN to learn to predict both properties, simultaniously.


## Operation object:
We now review the different options of *Operations* that *IGNNITION* allows, and which can be used in many of the parts of the *GNN* (e.g., message function, update function, readout function...). All these possible operations are:<br>

- [Operation 1: product](#operation-1-product)<br>
- [Operation 2: neural_network](#operation-2-neural_network)<br>
- [Operation 3: pooling](#operation-3-pooling)


---
### - Operation 1: product
This operation will perform the product of two different inputs. Let us go through the different parameters that we can tune to customize this operation.<br>

- [Parameter: input](#parameter-type)<br>
- [Parameter: output_name](#parameter-nn_name)
- [Parameter: type_product](#parameter-type_product)

---

#### Parameter: input
**Description:** Defines the set of inputs to be fed to this operation.
**Allowed values:** Array of two strings, defining the two inputs of the *product operation*.

Notice that if a string from the input references a feature from the dataset, the name must always be preceeded by a # symbol. This will indicate *IGNNITION* that such keyword references a value present in the dataset.

---

#### Parameter: output_name
**Description:** Defines the name by which we can reference the output of this operation if successive operations.

**Allowed values:** String

---

#### Parameter: type_product
**Description:** Defines the type of product that we use (e.g., element-wise, matrix multiplication, dot-product)

**Allowed values:** [dot_product, element_wise, matrix_mult]

Let us explain in more detail what each of the following keywords stands for:<br>
- [Option 1: dot_product](#option-1-dot_product)<br>
- [Option 2: element_wise](#option-2-element_wise)<br>
- [Option 3: matrix_mult](#option-3-matrix_mult)

---

##### Option 1: dot_product
**Description:** Computes the dot product between two inputs *a* and *b*. Note that if the inputs are two arrays *a = (a_1, a_2, ... , a_k)* and *b = (b_1, ,b_2, ... , b_k)*, then the dot product is applied to *a_i* and *b_i* respectively.
**Allowed values:** String. Name of an entity or output of a previous operation. 

Below we show an example of a readout function which first computes the *dot_product* between nodes of type *entity1* and *entity2*, respectively. Then, the result of each of these operations are passed to a *Neural Network* that compute the prediction.

```yaml
readout:
- type: product
  type_product: dot_product
  input: [entity1, entity2]
  nn_name: my_network1
  output_label: output1
- type: neural_network
  input: [output1, entity2]
  nn_name: my_network2
  output_label: [$my_label]
```

---

##### Option 2: element_wise
**Description:** Computes the element-wise multiplication between two inputs *a* and *b*. Note that if the inputs are two arrays *a = (a_1, a_2, ... , a_k)* and *b = (b_1, ,b_2, ... , b_k)*, then the element-wise multiplication is applied to *a_i* and *b_i* respectively.
**Allowed values:** String. Name of an entity or output of a previous operation. 

Below we show an example of a readout function which first computes the *element_wise* multiplication between nodes of type *entity1* and *entity2*, respectively. Then, the result of each of these operations are passed to a *Neural Network* that compute the prediction.

```yaml
readout:
- type: product
  type_product: dot_product
  input: [entity1, entity2]
  nn_name: my_network1
  output_label: output1
- type: neural_network
  input: [output1, entity2]
  nn_name: my_network2
  output_label: [$my_label]
```
---

##### Option 3: matrix_mult
**Description:** Computes the matrix multiplication between two inputs *a* and *b*. Note that if the inputs are two arrays *a = (a_1, a_2, ... , a_k)* and *b = (b_1, ,b_2, ... , b_k)*, then the matrix multiplication is applied to *a_i* and *b_i* respectively.
**Allowed values:** String. Name of an entity or output of a previous operation. 

Below we show an example of a readout function which first computes the *dot_product* between nodes of type *entity1* and *entity2*, respectively. Then, the result of each of these operations are passed to a *Neural Network* that compute the prediction.

---

### Operation 2: neural_network
Similarly to the neural_network operations used in the *message* or *update* function, we just need to reference the neural network to be used, and provide a name for the output.
Then, given some input \(a\) and a neural network that we define \(f\), this operation performs the following:

\(output\_name = f(a)\)

Below we show a code-snipped of what a *neural_network* operation would look like, and we present afterward each of its possible options. This neural network takes as input all the states of the nodes of type *entity1*, and pass them (separetely) to our *NN* named *my_network*. Finally it stores the results of these operations in *my_output*.

```yaml
- type: neural_network
  input: [entity1]
  nn_name: my_network
  output_name: my_output
```

We can now review in more depth each of its available parameters:<br>
- [Parameter: nn_name](#parameter-nn_name)<br>
- [Parameter: output_name](#parameter-output_name)

---

#### Parameter: input
**Description:** Defines the set of inputs to be fed to this operation.
**Allowed values:** Array of strings. If this neural network is part of the readout, you can use *entity1_initial_state* to reference the initial values of the hidden-states of *entity1*. Note that *entity1* can be replaced for any entity name of the model.

An important consideration is that all the strings in the input that reference a features --that is present in the dataset-- must be proceeded by a # symbol. This will indicate *IGNNITION* that such keyword references a value from the dataset.

---


#### Parameter: nn_name
**Description:** Name of the neural network \(f\), which shall then used to define its actual architecture in [Step 4](#step-4-internal-neural-networks).

**Allowed values:** String. This name should match the one from one of the neural networks defined.

---

#### Parameter: output_name
**Description:** Defines the name by which we can reference the output of this operation, to be then used in successive operations.

**Allowed values:** String

An example of the use of this operation is the following *message* function (based on a pipe-line of two different operations):

```yaml
message:
- type: neural_network
  input: [entity1]
  nn_name: my_network1
  output_name: my_output
  
- type: neural_network
  input: [my_output]
  nn_name: my_network2
```

With this, hence, we apply two successive neural networks, which is just a prove of some of the powerfull operations that we can define.

---

### Operation 3: pooling
The use of this operation is key to make global predictions (over the whole graph) instead of node predictions. This allows to take a set of input \(a_1, ... , a_k\) and a defined function \(g\), to obtain a single resulting output. This is:

\(output\_name = g(a_1, ..., a_k)\)

For this, we must define, as usual, the *output_name* field, where we specify the name for the output of this operation. Additionally, we must specify which function \(g\) we want to use. Let us see how this operation would look like if used to define a *readout* function to make global predictions over a graph. In this example we again define a pipe-line of opeartions, first of all by pooling all the nodes of type *entity1* together into a single representation (which is stored in my_output. Then we define a neural network operation which takes as input this pooled representation and applies it to a *NN* which aimy to predict our label *my_label*.

```yaml
readout:
- type: pooling
  type_pooling: sum/mean/max
  input: [entity1]
  output_name: my_output
  
- type: neural_network
  input: [my_output]
  nn_name: readout_model
  output_label: [$my_label]
```

Again, we now present the new keyword that is charactheristic from this specific operation:

#### Parameter: type_pooling:
**Description:** This field defines the pooling operation that we want to use to reduce a set of inputs \(a_1, ... , a_k\) to a single resulting output.

**Allowed values:** Below we define the several values that this field *type_pooling* can take:

Let us now explain in depth what each of the possible types of pooling that *IGNNITION* currently supports: <br>
- [Option 1: sum](#option-1-sum)<br>
- [Option 2: max](#option-2-max)<br>
- [Option 3: mean](#option-3-mean)
---

##### Option 1: sum
This operations takes the whole set of inputs \(a_1, ... , a_k\), and sums them all together.

\(output\_name = \sum(a_1, ... , a_k)\)

```yaml
- type: pooling
  type_pooling: sum
  input: [entity1]
```

---

##### Option 2: max

This operations takes the whole set of inputs \(a_1, ... , a_k\), and outputs the its max.

\(output\_name = \max(a_1, ... , a_k)\)

```yaml
- type: pooling
  type_pooling: max
  input: [entity1]
```
---

##### Option 3: mean

This operations takes the whole set of inputs \(a_1, ... , a_k\), and calculates their average.

\(output\_name = \frac{1}{k} \sum(a_1, ... , a_k)\)

```yaml
- type: pooling
  type_pooling: mean
  input: [entity1]
```

## Step 4: Neural Network architectures
In this section we define the architecture of the neural networks that we refenced in all the previous sections. For this, we just need to define an array of [Neural Network object](neural-network-object). Note that we will use the very same syntax to define either *Feed-forward NN* or *Recurrent NN*. Let us describe what a [Neural Network object](neural-network-object) looks like:

### Neural Network object
A Neural Network object refers to the architecture of an specific Neural Network. To do so, we must define two main fields, these being *nn_name* and *nn_architecture* which we define below. 

We can now review in more depth each of its available parameters:<br>
- [Parameter: nn_name](#parameter-nn_name)<br>
- [Parameter: nn_architecture](#parameter-nn_architecture)

---

#### Parameter: nn_name
**Description:** Name of the Neural Network. 

**Accepted values:** String. This name must match all the references to this Neural Network from all the previous sections (e.g., the name of the *NN* of the previous example would be *my_neural_network*)

---

#### Parameter: nn_architecture
**Description:** Definition of the actual architecture of the *NN*.

**Accepted values:** Array of Layer objects (e.g., a single *Dense* layer for the previous *NN*)


Let us now, for sake of the explanation, provide a simple example of how a *Neural Network* object can be defined:

```yaml
neural_networks:
- nn_name: my_neural_network
  nn_architecture:
  - type_layer: Dense
    units: readout_units
```


### Layer object
To define a Layer, we rely greatly on the well-known [tf.keras library](https://www.tensorflow.org/api_docs/python/tf/keras/layers). In consequence, we just require the user to define the following field. 

---

#### Parameter: type_layer
**Description:** Here we must indicate the type of layer to be used. Please writte only the layers accepted by the [tf.keras.layers library](https://www.tensorflow.org/api_docs/python/tf/keras/layers) using the same syntax.

**Allowed values:** String. It must match a layer from the *tf.keras.layers library*

```yaml
- type_layer: Dense/Softmax/...
  ...
```

#### Other parameters
Additionally, the user can define any other parameter from the [tf.keras library](https://www.tensorflow.org/api_docs/python/tf/keras/layers) corresponding to the type of layer defined. Note that in many occasions, the user is in fact required to define layer specific attributes (e.g., the number of units when creating a Dense layers). Thus, please make sure to define all mandatory parameters, and then, additionally, define optional parameters if needed.

E.g., if we define a Dense layer, we must first define the required parameter *units* (as specified by Tensorflow). Then, we can also define any optional parameter for the Dense class (visit [documentation](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense)) such as the activation or the use of bias.

```yaml
- type_layer: Dense
  units: 32
  activation: relu
  use_bias: False
```

