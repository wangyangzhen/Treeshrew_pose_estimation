# Treeshrew pose estimation <img src="/README/logo.jpg" width="320px" align="right" />
We present a deep learning approach to achieve markerless pose estimation and recognize multiple spontaneous behaviors of tree shrews, including drinking, eating, resting, and staying in the dark house. This high-throughput approach can monitor the home-cage activities of tree shrews simultaneously over an extended period.This study provides an efficient tool to quantify and understand the natural behaviors of tree shrews.

## Requirements: 
* Windows on 64-bit x86 
* NVIDIA GPU (Tested on Nvidia GeForce RTX 3090)
* Python 3.8

## Get start: 
To run example:
```bash
python main.py --repo_path ./Treeshrew_pose_estimation --video_path ./demo_movie --video_name demo_movie.mp4
```
The demo video can be easily analyzed, and output the pose estimation of tree shrew, including animal's Ear_left, Ear_right, Nose, Center, Lateral_left, Lateral_right,Tail_base,Tail_end.

## Behavior annotation: 
To define the behavior states of animal, please use video_annotation.py. First input custom defined behavior in the top input box, press "next" to save. After definition of behaviors, select the video that you want to label. Select the behavior you want to annotate the frame. Press the Space bar on the keyboard to save
the annotation and load the next frame of video. The unlabeled part of video will not be used for further analyze.

## Behavior prediction using RandomForest Classifier: 
After video analyze and annotation, user can utilize RF_model.py to train the RandomForest Classifier model and evulate the model's performance.
