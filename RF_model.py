# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 17:15:48 2023

@author: yangzhen
"""


import h5py
from sklearn.ensemble import RandomForestClassifier
import joblib

# Read the 'demo_movieDLC_resnet50_treeshew_DLCJan1shuffle1_150000.h5' file
with h5py.File('demo_movieDLC_resnet50_treeshew_DLCJan1shuffle1_150000.h5', 'r') as file:
    x = file['data'][:]  # Assuming the dataset is named 'data'

# Read the 'label.h5' file
with h5py.File('label.h5', 'r') as file:
    y = file['labels'][:]  # Assuming the dataset is named 'labels'

# Create a RandomForestClassifier model
model = RandomForestClassifier()

# Train the model using the data
model.fit(x, y)

# Save the trained model
joblib.dump(model, 'demo_movie_RF.pkl')
