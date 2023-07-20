# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 17:15:48 2023

@author: yangzhen
"""


import argparse
import os
import pandas as pd
from deeplabcut import analyze_videos,create_labeled_video, plot_trajectories
from deeplabcut.pose_estimation_tensorflow.config import load_config
from deeplabcut.utils import auxiliaryfunctions
import numpy as np
parser = argparse.ArgumentParser(description='Tree shrew pose estimation')
parser.add_argument('--repo_path', type=str, default='D:/data/behavior/treeshew/code/',
                    help='path to repositorie)')
parser.add_argument('--video_path', type=str, default="D:/data/behavior/treeshew/movie/", metavar='N',
                    help='path to video file')
parser.add_argument('--video_name', type=str, default="demo_movie.mp4", metavar='N',
                    help='path to video file')

args = parser.parse_args()

project_path=os.path.join(args.repo_path,'project/config.yaml')
dlc_cfg = load_config(os.path.join(args.repo_path,'project/dlc-models/iteration-0/treeshew_DLCJan1-trainset95shuffle1/test/pose_cfg.yaml'))
cfg = auxiliaryfunctions.read_config(project_path)
analyze_videos(project_path, args.video_path, save_as_csv=True)
create_labeled_video(project_path, [os.path.join(args.video_path,args.video_name)], draw_skeleton=True)
plot_trajectories(project_path,[os.path.join(args.video_path,args.video_name)],showfigures=False)

#%%
Snapshots = np.array(
    [
        fn.split(".")[0]
        for fn in os.listdir(os.path.join(args.repo_path, "project/dlc-models/iteration-0/treeshew_DLCJan1-trainset95shuffle1/train"))
        if "index" in fn
    ]
)
increasing_indices = np.argsort([int(m.split("-")[1]) for m in Snapshots])
Snapshots = Snapshots[increasing_indices]
snapshotindex = cfg["snapshotindex"]
dlc_cfg["init_weights"] = os.path.join(
    "project/dlc-models/iteration-0/treeshew_DLCJan1-trainset95shuffle1/train", Snapshots[snapshotindex]
)
trainingsiterations = (dlc_cfg["init_weights"].split(os.sep)[-1]).split("-")[-1]
DLCscorer,_= auxiliaryfunctions.get_scorer_name(
    cfg,
    shuffle=1,
    trainFraction=cfg["TrainingFraction"][0],
    trainingsiterations=trainingsiterations,
    modelprefix=""
)
df=pd.read_hdf(os.path.join(args.video_path,os.path.splitext(args.video_name)[0]+DLCscorer+'.h5'))


