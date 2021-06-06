import os
import argparse
import torch
from glob import glob
from lib.lib_fun import prepare_segmentations, openpose_from_file

from model.mesh_generator import BaseNet, ShapeOptNet
from model.train_predict import  fine_tuning



def main(weights_dir, segmented_frame_dir, pose_keypoints_dir, opt_pose_steps, opt_shape_steps, out_dir):


    ## Load input
    segmentation_add = sorted(glob(os.path.join(segmented_frame_dir, '*.png')))
    pose_add = sorted(glob(os.path.join(pose_keypoints_dir, '*.json')))

    ## prepare inputs

    input_seg = [prepare_segmentations(seg).unsqueeze(0) for seg in segmentation_add]

    pose_keypoints = []
    face_keypoints = []
    for f in pose_add:

        j, f = openpose_from_file(f)

        assert(len(j) == 25)
        assert(len(f) == 70)

        pose_keypoints.append(torch.tensor(j, dtype=torch.float32))
        face_keypoints.append(torch.tensor(f, dtype=torch.float32))

    input_pose = pose_keypoints
    input_face = face_keypoints

    ## train and predict
    num_f = 8 # frame number
    base_model = BaseNet()
    model = ShapeOptNet(base_model)

    fine_tuning(model, base_model, weights_dir, input_seg, input_pose, input_face, num_f, opt_pose_steps, opt_shape_steps, out_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--segmented_frame_dir',
        default= "./input/segmentations/",
        help="Segmentation images directory")

    parser.add_argument(
        '--pose_keypoints_dir',
        default= "./input/keypoints",
        help="2D pose keypoints directory")

    parser.add_argument(
        '--pose_steps', '-p', default=80, type=int,
        help="Pose optimization steps")

    parser.add_argument(
        '--shape_steps', '-s', default=80, type=int,
        help="Shape optimization steps")

    parser.add_argument(
        '--weights_dir', '-w',
        default='weights/weights.pth',
        help='Model weights file (*.pth)')

    parser.add_argument(
        '--output_dir', '-out',
        default='./output',
        help='Output directory')

    args = parser.parse_args()
    main(args.weights_dir, args.segmented_frame_dir, args.pose_keypoints_dir, args.pose_steps, args.shape_steps, args.output_dir)
