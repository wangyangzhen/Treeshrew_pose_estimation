# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 13:49:27 2023

@author: yangzhen
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import cv2


# Step 1: Read the .h5 file
h5_file = 'path_to_your_h5_file.h5'
with h5py.File(h5_file, 'r') as f:
    # Get the coordinates of the features
    coordinates = f['df_with_missing']['table'][()]
    # Get the frame rate of the video
    frame_rate = f['config']['videotype']['framerate'][()]

# Step 2: Calculate the average motion speed for each feature point
num_frames = coordinates.shape[0]
num_features = coordinates.shape[1]
speeds = np.zeros((num_frames-1, num_features))
for i in range(num_frames-1):
    for j in range(num_features):
        dx = coordinates[i+1, j, 0] - coordinates[i, j, 0]
        dy = coordinates[i+1, j, 1] - coordinates[i, j, 1]
        distance = np.sqrt(dx**2 + dy**2)
        time = 1 / frame_rate
        speed = distance / time
        speeds[i, j] = speed

# Step 3: Generate velocity plots for each feature point
time_axis = np.arange(num_frames-1) / frame_rate
for i in range(num_features):
    plt.plot(time_axis, speeds[:, i], label='Feature {}'.format(i+1))
plt.xlabel('Time (s)')
plt.ylabel('Speed (pixels/s)')
plt.legend()
plt.show()

# Step 4: Generate heatmaps for each feature point
heatmap = np.zeros((coordinates.shape[2], coordinates.shape[3]))
for i in range(num_frames):
    x = coordinates[i, :, :, 0]
    y = coordinates[i, :, :, 1]
    heatmap[y, x] += 1
plt.imshow(heatmap, cmap='jet')
plt.colorbar()
plt.show()

# Step 5: Interactive analysis of local regions
image_file = 'path_to_your_video_first_frame.png'
image = cv2.imread(image_file)

# Draw circles and rectangles on the first frame
circles = []
rectangles = []

def draw_circles(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        circles.append((x, y))
        cv2.circle(image, (x, y), 10, (0, 0, 255), -1)

def draw_rectangles(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        rectangles.append((x, y))
        cv2.rectangle(image, (x-20, y-20), (x+20, y+20), (0, 255, 0), 2)

cv2.namedWindow('Image')
cv2.setMouseCallback('Image', draw_circles)

while True:
    cv2.imshow('Image', image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()

# Save the annotated image
annotated_image_file = 'path_to_save_annotated_image.png'
cv2.imwrite(annotated_image_file, image)

# Calculate average speed, time, and in/out counts for each feature point in each region
region_speeds = []
region_times = []
region_in_out_counts = []

for circle in circles:
    x, y = circle
    distances = np.sqrt((coordinates[:, :, :, 0] - x)**2 + (coordinates[:, :, :, 1] - y)**2)
    in_region = np.any(distances <= 10, axis=(1, 2))
    out_region = np.any(distances > 10, axis=(1, 2))
    region_speeds.append(np.mean(speeds[in_region]))
    region_times.append(np.sum(in_region) / frame_rate)
    region_in_out_counts.append((np.sum(in_region), np.sum(out_region)))

for rectangle in rectangles:
    x, y = rectangle
    distances = np.abs(coordinates[:, :, :, 0] - x) + np.abs(coordinates[:, :, :, 1] - y)
    in_region = np.any(distances <= 20, axis=(1, 2))
    out_region = np.any(distances > 20, axis=(1, 2))
    region_speeds.append(np.mean(speeds[in_region]))
    region_times.append(np.sum(in_region) / frame_rate)
    region_in_out_counts.append((np.sum(in_region), np.sum(out_region)))

# Save the results as .h5 file
results_file = 'path_to_save_results.h5'
with h5py.File(results_file, 'w') as f:
    f.create_dataset('region_speeds', data=region_speeds)
    f.create_dataset('region_times', data=region_times)
    f.create_dataset('region_in_out_counts', data=region_in_out_counts)
