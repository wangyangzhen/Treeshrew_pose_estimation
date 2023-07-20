# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 17:15:48 2023

@author: yangzhen
"""

import pandas as pd
import h5py
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib



file_X = './demo_movieDLC_resnet50_treeshew_DLCJan1shuffle1_100000.h5'

file_Y = '.movie/demo_movie.h5'

coor = pd.read_hdf(file_X, "df_with_missing").values

with h5py.File(file_Y, 'r') as file:
    video_label = file['video_label'][:]

# Prepare the data for training
X_train = []
y_train = []
for i in range(len(video_label)):
    if video_label[i] != -1:
        # Include the data in training set if the label is not -1
        X_train.append(coor[i])
        y_train.append(video_label[i])

# Create and train the random forest classifier
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)

# Calculate the accuracy on the training set
y_train_pred = rf_classifier.predict(X_train)
accuracy = accuracy_score(y_train, y_train_pred)
print("Training Accuracy:", accuracy)

# Save the trained model
model_file = 'path_to_save_model.pkl'
joblib.dump(rf_classifier, model_file)
print("Model saved successfully.")