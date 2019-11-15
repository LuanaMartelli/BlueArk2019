# Read CSV file
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

data = pd.read_csv('./dataset/myfile.csv', names=['x1', 'x2', 'y'])

x1 = data['x1'].values
x2 = data['x2'].values
y = data['y'].values

X = np.c_[x1, x2]
X = np.c_[np.ones(X.shape[0]), X]


