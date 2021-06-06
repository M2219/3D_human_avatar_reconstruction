#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 11:38:33 2020

@author: mahmoud

This script extract frames and masks from a video.mp4
and stores masks from a directory in a compressed hdf5 file.

"""

import os
from glob import glob
import numpy as np
import h5py
from tqdm import tqdm
from cv2 import cv2
import skimage.exposure



def mask_extract(frame_paths, mask_results_path, h_th, s_th, val_th):

    """
    input: frames
    output: masks.png

    h_th : hue threshold
    s_th : saturation threshold
    val_th : value threshold

    """

    count = 0

    for img_add in frame_paths:


        # mask extraction
        frame0 = cv2.imread(img_add)

        out_mask = mask_genrator(frame0, h_th, s_th, val_th)

        # save mask
        cv2.imwrite(mask_results_path +"%05d" % count+'.png', out_mask)

        count += 1


def mask_genrator(frame0, h_th, s_th, val_th):

        hsv = cv2.cvtColor(frame0, cv2.COLOR_BGR2HSV)
        hsv2d = hsv.transpose(2, 0, 1).reshape(3, -1)
        hue = hsv2d[0][:]
        satu = hsv2d[1][:]
        val = hsv2d[2][:]

        # eliminate background using hsv
        lower_green = np.array([hue.mean() - h_th, satu.mean() - s_th, val.mean() - val_th])
        upper_green = np.array([hue.mean() + h_th, satu.mean() + s_th, val.mean() + val_th])

        mask = cv2.inRange(hsv, lower_green, upper_green)
        mask = 255 - mask

        # apply morphology opening to mask
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # antialias mask
        # mask = cv2.GaussianBlur(mask, (0,0), sigmaX=3, sigmaY=3, borderType = cv2.BORDER_DEFAULT)
        mask = skimage.exposure.rescale_intensity(mask, in_range=(127.5,255), out_range=(0,255))

        return mask


if __name__ == '__main__':


    src_path = "/app/frame_and_mask_extractor/output/selected_frames/"
    # src_path = "/home/mahmoud/Desktop/hoopad/frame_and_mask_extractor/frames_paper"
    frame_path = sorted(glob(os.path.join(src_path, '*.jpg')))


    # mask and frame extraction
    mask_path = os.path.join("/app/frame_and_mask_extractor/output/selected_masks/")


    if not os.path.exists(mask_path):
        os.makedirs(mask_path)


    hue_thresh = 10
    sat_thresh = 50
    val_thresh = 50

    mask_extract(frame_path, mask_path, hue_thresh, sat_thresh, val_thresh)
    
    
    
    
    
    
    
    
    
    
    
