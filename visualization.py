# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 17:15:48 2023

@author: yangzhen
"""

import numpy as np
import h5py
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter




# Read the .h5 file
h5_file = h5py.File('your_file.h5', 'r')

# Read the feature point data
body_parts = ['body_part_1', 'body_part_2', 'body_part_3', 'body_part_4', 'body_part_5', 'body_part_6', 'body_part_7', 'body_part_8']
feature_points = []
for body_part in body_parts:
    feature_points.append(h5_file['df_with_missing']['table'][body_part]['x'][:])
    feature_points.append(h5_file['df_with_missing']['table'][body_part]['y'][:])
feature_points = np.array(feature_points)

# Calculate the average motion speed for each feature point
frame_rate = h5_file['fps'][()]
dist_per_pixel = 0  # Length represented by each pixel (in centimeters)
# Display the first frame of the video and let the user draw a line
first_frame = cv2.cvtColor(cv2.imread('your_video_frame.jpg'), cv2.COLOR_BGR2RGB)
plt.imshow(first_frame)
plt.show()
# User inputs the length of the line in centimeters
line_length_cm = float(input("Enter the length of the line (in centimeters): "))
# Calculate the length represented by each pixel
line_length_pixels=1
dist_per_pixel = line_length_cm / line_length_pixels

# Calculate the velocity for each feature point
num_frames = feature_points.shape[1]
velocities = np.zeros((len(body_parts), num_frames-1))
for i in range(len(body_parts)):
    for j in range(num_frames-1):
        dx = feature_points[i*2, j+1] - feature_points[i*2, j]
        dy = feature_points[i*2+1, j+1] - feature_points[i*2+1, j]
        distance = np.sqrt(dx**2 + dy**2)
        velocities[i, j] = distance / dist_per_pixel * frame_rate

# Generate velocity plots for each feature point
time = np.arange(num_frames-1) / frame_rate
for i in range(len(body_parts)):
    plt.plot(time, velocities[i], label=body_parts[i])
plt.xlabel('Time (s)')
plt.ylabel('Velocity')
plt.legend()
plt.show()

# Generate heatmaps for each feature point
heatmap_data = np.zeros((first_frame.shape[0], first_frame.shape[1]))
for i in range(num_frames):
    x = feature_points[::2, i].astype(int)
    y = feature_points[1::2, i].astype(int)
    heatmap_data[y, x] += 1
heatmap_data = gaussian_filter(heatmap_data, sigma=5)  # Apply Gaussian smoothing
plt.imshow(heatmap_data, cmap='jet')
plt.colorbar()
plt.show()
