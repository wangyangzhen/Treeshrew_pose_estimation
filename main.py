# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 17:15:48 2023

@author: Admion
"""

import deeplabcut
import argparse
import os
import pandas as pd


parser = argparse.ArgumentParser(description='Tree shrew pose estimation')
parser.add_argument('--repo_path', type=str, default='D:/data/behavior/treeshew/code/',
                    help='path to repositorie)')
parser.add_argument('--video_path', type=str, default="D:/data/behavior/treeshew/movie/", metavar='N',
                    help='path to video file')
parser.add_argument('--video_namme', type=str, default="demo_movie.mp4", metavar='N',
                    help='path to video file')

args = parser.parse_args()

project_path=os.path.join(args.repo_path,'project/config.yaml')
deeplabcut.analyze_videos(project_path, args.video_path, save_as_csv=True)
#%%
deeplabcut.create_labeled_video(project_path, [os.join.path(args.video_path,args.video_namme)], draw_skeleton=True)
deeplabcut.plot_trajectories(project_path,[os.join.path(args.video_path,args.video_namme)],showfigures=True)

df=pd.read_hdf(os.join.path("D:/data/behavior/treeshew/movie/demo_movieDLC_resnet50_treeshew_DLCJan1shuffle1_150000.h5"))

