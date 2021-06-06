#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 11:38:33 2020

@author: mahmoud

This script extract frames from a video.mp4

"""

import os
from glob import glob
from cv2 import cv2
from os import listdir
from os.path import isfile, join
import json
import numpy as np


def resize_f(imgs):

    width_per = (1080/imgs.shape[1])
    height_per = (1080/imgs.shape[0])
    width = int(imgs.shape[1] *  width_per)
    er_w = abs(imgs.shape[1] - width)
    height = int(imgs.shape[0] *  height_per)
    er_h = abs(imgs.shape[1] - height)
    dim = (width + er_w, height + er_h)
    # resize image
    img1 = cv2.resize(imgs, dim, interpolation=cv2.INTER_AREA)
    return img1


def video_orientation(vid_path, first_frame_path, src_path):

    # Path to video file
    vidobj = cv2.VideoCapture(vid_path)

    success, image = vidobj.read()


    cv2.imwrite(first_frame_path + "first_frame.jpg", image)

    os.system('/app/openpose/openpose-1.5.1/build/examples/openpose/openpose.bin --image_dir ' + src_path + '/data/first_frame \
--display 0  --render_pose 0 --write_json ' + src_path + '/data/first_frame keypoints \
-net_resolution="368x368" -number_people_max 1 -model_folder /app/openpose/openpose-1.5.1/models')


    f = open(first_frame_path + "first_frame_keypoints.json",)
    keyp = json.load(f)

    key_2d = (np.asarray(keyp["people"][0]["pose_keypoints_2d"]))

    key_2d_np = np.concatenate(key_2d.reshape(-1, 1, 3))

    key_com = key_2d_np[20, :] - key_2d_np[23, :]
    
    ax = np.argmax(key_com)
    
    if ax == 0:

        if key_2d_np[20, :][1] > key_2d_np[8, :][1]:

            orientation = None
            orientation_str = "Head is up"

        else: 

            orientation = cv2.ROTATE_180
            orientation_str = "Head is bottom"

    else:

        if key_2d_np[20, :][0] > key_2d_np[8, :][0]:

            orientation = cv2.ROTATE_90_CLOCKWISE
            orientation_str = "Head is left"
        else:

            orientation = cv2.ROTATE_90_COUNTERCLOCKWISE
            orientation_str = "Head is right"

    print(orientation_str)

    return orientation
    # Saves the frames with frame-count


def frame_extract(vid_path, frame_results_path, oreintation):

    """
    input: video.mp4
    output: frames.hpg

    """
    # Path to video file
    vidobj = cv2.VideoCapture(vid_path)

	# Used as counter variable
    count = 0

	# checks whether frames were extracted
    success = 1

    while success==1:

		# vidObj object calls read
        success, image = vidobj.read()

        #rotation

        if oreintation is not None:

            image = cv2.rotate(image, oreintation) 

		# Saves the frames with frame-count

   
        if success == 1:

            # save frame
            image = resize_f(image)
            cv2.imwrite(frame_results_path + "%05d.jpg" % count, image)

            count += 1



if __name__ == '__main__':

    # mask and frame extraction

    src_path = "/allproj/hoopad/hoopad-oct-2020-31/frame_and_mask_extractor"



    video_name = [f for f in listdir("./input") if isfile(join("./input", f))]
    print(video_name)
    if len(video_name) != 1:
        print("Please use one video file")
        exit() 

    video_path = os.path.join(src_path + "/input/" + video_name[0])


    frame_path = os.path.join(src_path + "/data/frames/")

    first_frame_path = os.path.join(src_path + "/data/first_frame/")

    if not os.path.exists(frame_path):
        os.makedirs(frame_path)

    if not os.path.exists(first_frame_path):
        os.makedirs(first_frame_path)

    oreintation = video_orientation(video_path, first_frame_path, src_path)

    frame_extract(video_path, frame_path, oreintation)
