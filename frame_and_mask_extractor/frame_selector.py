#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 11:38:33 2020

@author: mahmoud

This script extract frames and masks from a video.mp4
and stores masks from a directory in a compressed hdf5 file.

"""
from __future__ import print_function
from __future__ import division
import cv2 as cv2
import os
from glob import glob
import numpy as np
import h5py
from tqdm import tqdm
from cv2 import cv2
from shutil import copyfile
from mask_extractor import mask_genrator
import argparse
import json


def return_X_keypoint(src_path, frame):

    ref_point = 1
    with open(src_path + "%05d_keypoints.json" % (frame),) as f:
        data = json.load(f)

    people = data['people'][0]
    keyp = np.asarray(people["pose_keypoints_2d"])
    keypoints = keyp.reshape([-1, 3])

    X = keypoints[:, 0]
    xf = X[ref_point]

    return X, xf


def find_oreintation(src_path, num_frame):

    oreintation = np.ones(num_frame)
    oreintation_prev = np.array([-1])
    oreintation_face_start = []
    oreintation_face_end = []

    for indx in range(0, num_frame, 1):

        X, xf = return_X_keypoint(src_path, indx)

        # face
        if all(X) == True:
            oreintation[indx] = 0

        if (oreintation[indx] == 0) and (oreintation_prev != 0):
            oreintation_face_start.append(indx)
        if (oreintation[indx] != 0) and (oreintation_prev == 0):
            oreintation_face_end.append(indx)


        oreintation_prev = oreintation[indx]

    number_of_rotation = min([len(oreintation_face_start), \
                             len(oreintation_face_end)])

    oreintation_face_end.append(num_frame - 1)

    return oreintation, (oreintation_face_start, oreintation_face_end), number_of_rotation


def find_best_face(src_path, start_frame, end_frame):

    min_sym = np.inf
    for indx in range(start_frame, end_frame + 1):

        X, xf = return_X_keypoint(src_path, indx)

        error = (abs(X[2] - xf) - abs(X[5] - xf)) ** 2
        error = error + (abs(X[3] - xf) - abs(X[6] - xf)) ** 2
        error = error + (abs(X[4] - xf) - abs(X[7] - xf)) ** 2

        if error < min_sym:
            min_sym = error
            ind_sym = indx

    return ind_sym

def find_best_alpha_frame(src_path, original_frame, start_frame, end_frame, alpha):

    #keypoints_ind = [2, 3, 4, 5, 6, 7, 9, 10, 12, 13]
    keypoints_ind = [2, 5, 9, 12]

    X, xf = return_X_keypoint(src_path, original_frame)

    x_rot_in_org = xf + (X[keypoints_ind] - xf) * np.cos(alpha)

    minerror = np.inf

    for i in range(int(start_frame), int(end_frame) + 1):


        X2, xf2 = return_X_keypoint(src_path, i)

        ## Two frames for comparison
        x_rot_in_com_org = x_rot_in_org + (xf2 - xf)
        x_com = X2[keypoints_ind]

        x_rot_in_com_org[x_com == 0] = 0

        error = np.sum((x_rot_in_com_org - x_com) ** 2)

        if error < minerror:

            minerror = error
            indmin = i

    return indmin


def find_all_selected_frame(src_path, first_face_frame, last_face_frame, num_output_frame):

    original_frame = first_face_frame

    start_frame = first_face_frame
    end_frame = last_face_frame
    alpha = np.pi
    back_frame = find_best_alpha_frame(src_path, original_frame, start_frame, end_frame, alpha)

    selected_frame = np.zeros(num_output_frame)

    for alpha_cnt in range(0, num_output_frame):

        alpha = alpha_cnt * 2 * np.pi / num_output_frame
        if alpha <= np.pi: 

            start_frame = first_face_frame
            end_frame = back_frame

        else:

            start_frame = back_frame
            end_frame = last_face_frame

        selected_frame[alpha_cnt] = find_best_alpha_frame(src_path, original_frame, start_frame, end_frame, alpha)

    return selected_frame



def mask_extractor(frame_paths, selected_frame_paths, selected_mask_paths, selected_frame, h_th, s_th, val_th):

    """
    input: frames
    output: masks and selected frames

    """

    total_num_frame = len(selected_frame)

    for i in selected_frame.astype(int):

        frame_name = os.path.basename(frame_path[i])
        copyfile(frame_path[i], selected_frame_paths + frame_name)

        # mask extraction
        frame0 = cv2.imread(frame_path[i])
        out_mask = mask_genrator(frame0, h_th, s_th, val_th)

        # save mask
        cv2.imwrite(selected_mask_paths + frame_name, out_mask)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'hue_thresh', default=10, type=int,
        help="hue_thresh")

    parser.add_argument(
        'sat_thresh', default=50, type=int,
        help="sat_thresh")

    parser.add_argument(
        'val_thresh', default=50, type=int,
        help="val_thresh")

    parser.add_argument(
        'rotation', default=2, type=int,
        help="selected rotation number, {1, 2, 3}")

    parser.add_argument(
        'num_output_frame', default=8, type=int,
        help="number of output frames")



    args = parser.parse_args()

    h_th = args.hue_thresh
    s_th = args.sat_thresh
    val_th = args.val_thresh
    rotation = args.rotation

    num_output_frame = args.num_output_frame

    src_path = "/app/frame_and_mask_extractor/"

    keypoint_path = src_path + "data/keypoints/"

    frame_path = sorted(glob(os.path.join(src_path, 'data/frames/*.jpg')))
    num_frame = len(frame_path)


    # mask and frame extraction
    selected_frame_paths = os.path.join("/app/frame_and_mask_extractor/output/selected_frames/")
    selected_mask_paths = os.path.join("/app/frame_and_mask_extractor/output/selected_masks/")

    if not os.path.exists(selected_frame_paths):
        os.makedirs(selected_frame_paths)

    if not os.path.exists(selected_mask_paths):
        os.makedirs(selected_mask_paths)

    oreintation, oreintation_face, number_of_rotation = find_oreintation(keypoint_path, num_frame)

    best_face = np.zeros(number_of_rotation + 1)
    print("number of rotation", number_of_rotation)
    print("orientation face", oreintation_face)
    print("orientation", oreintation)
    for rot_num in range(0, number_of_rotation):

        start_frame_face = oreintation_face[0][rot_num]
        end_frame_face = oreintation_face[1][rot_num]

        best_face[rot_num] = find_best_face(keypoint_path, start_frame_face, end_frame_face)


    selected_rotation_number = rotation - 1 # 0, 1, 2 for 3 rotation

    selected_frame = find_all_selected_frame(keypoint_path, best_face[selected_rotation_number], best_face[selected_rotation_number + 1], num_output_frame)

    print(selected_frame)

    mask_extractor(frame_path, selected_frame_paths, selected_mask_paths, selected_frame, h_th, s_th, val_th)

