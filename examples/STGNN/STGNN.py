import pandas as pd
import numpy as np
import networkx as nx
import scipy.sparse as sp
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import transformerblock as tran
import GCN
import os
import matplotlib.pyplot as plot

def predictor_layer(name="Predictor"):
    predictor_layers =[]
    predictor_layers.append(keras.layers.Dense(64,activation='relu'))
    predictor_layers.append(keras.layers.Dense(32,activation='relu'))
    predictor_layers.append(keras.layers.Dense(1))

    return keras.Sequential(predictor_layers, name=name)

def create_ffn(hidden_units, dropout_rate, name=None):
    fnn_layers = []
    fnn_layers.append(layers.BatchNormalization())

    for units in hidden_units:
        fnn_layers.append(layers.Dense(units, activation=tf.nn.relu))
        fnn_layers.append(layers.Dropout(dropout_rate))
        
    fnn_layers.append(layers.Dense(units, activation=tf.nn.relu))
    return keras.Sequential(fnn_layers, name=name)

class STGNN(tf.keras.Model):
    def __init__(
        self,
        graph_info, #matriu d'adjecencia del graph
        hidden_units, #quantes features estarem tractant per timestep i sensor
        aggregation_type="sum",
        combination_type="gru",
        dropout_rate=0.2,
        normalize=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Unpack graph_info to three elements: node_features, edges, and edge_weight.
        edges, edge_weights = graph_info
        self.edges = edges
        self.edge_weights = edge_weights
        # Set edge_weights to ones if not provided.
        if self.edge_weights is None:
            self.edge_weights = tf.ones(shape=edges.shape[1])
        # Scale edge_weights to sum to 1.
        self.edge_weights = self.edge_weights / tf.math.reduce_sum(self.edge_weights)
        self.aggregation_type = aggregation_type
        self.combination_type = combination_type
        self.normalize = normalize
        self.hidden_units = hidden_units


        # Create a process layer.
        self.preprocess = create_ffn(hidden_units, dropout_rate, name="preprocess")
        # Create the first GraphConv layer.
        self.conv1 = GCN.GraphConvLayer(
            hidden_units,
            dropout_rate,
            aggregation_type,
            combination_type,
            normalize,
            name="graph_conv1",
        )

        self.grucell = tf.keras.layers.GRUCell(64,name="Gru_temporal")

        # Create the temporaL BLOCK
        self.tbloc = tran.TransformerBlock(
            embed_dim=64, 
            num_heads=4, 
            ff_dim=64
        )

        self.pool = keras.layers.GlobalAveragePooling1D()
        
        self.predictor = predictor_layer()
        # la prediction layer es una multi feed-forward network per predir veloocitat transit

    
    

    def call(self, input_node_features_all):
        old_x = input_node_features_all[0] #[325]
        old_x = tf.expand_dims(old_x,axis=1)
        old_x = self.preprocess(old_x)
        
        x_full= tf.expand_dims(old_x,axis=1)

        for input_node_features in input_node_features_all[1:] :

            tf.autograph.experimental.set_loop_options(
            shape_invariants=[(x_full, tf.TensorShape([x_full.shape[0],None,x_full.shape[2]]))]
            )

            # Preprocess the node_features to produce node representation
            # x = tf.reshape(input_node_features_all,[input_node_features_all.shape[1],input_node_features_all.shape[0]])
            x =tf.expand_dims(input_node_features,axis = 1)
            x = self.preprocess(x)
            # Apply the first graph conv layer.
            x1 = self.conv1((x, self.edges, self.edge_weights))
            # Skip connection.
            x = x1 + x
            # Apply the second graph conv layer.
            # Postprocess node embedding.
            x, x2 = self.grucell(x,old_x) 
            old_x = x2
            x_full = tf.concat([x_full,tf.expand_dims(old_x,axis=1)],1)
        

        
        x_full = tf.convert_to_tensor(x_full,tf.float32)
        # sortiran bastant més valors del transformer, es pot provar amb i o sense transformer 
        res = self.tbloc(x_full) #hauria de treure 325, 12, 64
        res = self.pool(res) #[325,64]
        x = self.predictor(res) # hauria de treure 325, 12, 1 SEWMPRE una sortida
        # return tf.squeeze(tf.transpose(x)) #es treu el valor amb forma (325,) que es equivalent a [325,1] amb feina extra
        return x
