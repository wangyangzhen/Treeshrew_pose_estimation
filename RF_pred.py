# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 17:15:48 2023

@author: yangzhen
"""


import pandas as pd
import numpy as np
import h5py
import pickle
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Load the trained model from 'demo_movie_RF.pkl' file
with open('demo_movie_RF.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the data from the '.h5' file
data = h5py.File('your_data.h5', 'r')
x = np.array(data['your_data'])

# Make predictions using the trained model
y = model.predict(x)

# Save the predictions as a new '.h5' file
with h5py.File('predictions.h5', 'w') as file:
    file.create_dataset('predictions', data=y)

# Generate and save the Gantt chart for the predictions
plt.figure(figsize=(10, 6))
plt.bar(range(len(y)), y)
plt.xlabel('Data Point')
plt.ylabel('Prediction')
plt.title('Prediction Gantt Chart')
plt.savefig('gantt_chart.png')
