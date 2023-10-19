import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd

dataset = pd.read_csv("examples/STGNN/datasets/PEMS-BAY.csv")
dataset_N = dataset.to_numpy()
pred = pd.read_csv("results_v2.csv")

truth = dataset_N[2006][1:]
pred = pred.to_numpy()[0]
mae = abs(truth-pred)
print(np.mean(mae))
aux = abs((truth-pred)/truth)
print(np.mean(100*aux))

mse = np.mean(np.square(truth-pred))
rmse = np.sqrt(mse)
print(rmse)