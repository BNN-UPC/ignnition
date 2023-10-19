import networkx as nx
import pandas as pd
import json
import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("examples/STGNN/datasets/PEMS-BAY.csv")
matrix = pd.read_pickle('examples/STGNN/datasets/adj_mx_PEMS-BAY.pkl')

nom_nodes=matrix[0]
neighbours=matrix[1]
adjacencies=matrix[2]

id = np.identity(325)

adjacencies = adjacencies - id
# G = nx.read_multiline_adjlist('examples/STGNN/datasets/adj_mx_PEMS-BAY.pkl')
preds = pd.read_csv("results.csv")
preds_N = preds.to_numpy()
print(len(preds_N))
data =[]
for i in range(97):

    G = nx.DiGraph(adjacencies)

    hidden_value = df
    hidden_A  = hidden_value.to_numpy()

    for x in G:
        for j in range(3): #primer apred era començant 2000 - 2003 
            # G.nodes[x]["Q_value_"+str(j)]=np.random.random()*100
            G.nodes[x]["Q_value_"+str(j)]= preds_N[i+j][x]
            # G.nodes[x]["Q_value_"+str(j)]=hidden_A[i+j+2000][x+1]
        G.nodes[x]["val_value"]=hidden_A[i+2006][x]
        

    nx.set_node_attributes(G,'sensor','entity')
    json_data = nx.json_graph.node_link_data(G)

    data.append(json_data)

with open('examples/STGNN/datasets/predict/data_big_predict.json','w') as json_file:
    json.dump(data,json_file)
