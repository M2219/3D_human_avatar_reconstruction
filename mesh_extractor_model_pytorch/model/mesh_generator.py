import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from lib.lib_fun import laplacian_tensor
from lib.gcl_torch import Gcl, sparse_mat_to_sorted_sparse, chebyshev_polynomials
import os
import pickle as pkl
from smpl.batch_lbs_torch import batch_rodrigues
from smpl.batch_smpl_torch import SMPL
from smpl.joints_torch import joints_body25, face_landmarks
from render.render_torch import render

torch.autograd.set_detect_anomaly(True)
torch.cuda.empty_cache()

if torch.cuda.is_available() == True:
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    print("cuda is available")
else:
    device = torch.device("cpu")
    print("cuda is not available")



class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()

        self.conv_1 = nn.Conv2d(3, 8, kernel_size=(3, 3), stride=(2, 2))
        self.m_pool_1 = nn.MaxPool2d((2, 2))

        self.conv_2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1))
        self.m_pool_2 = nn.MaxPool2d((2, 2))

        self.conv_3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1))
        self.m_pool_3 = nn.MaxPool2d((2, 2))

        self.conv_4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
        self.m_pool_4 = nn.MaxPool2d((2, 2))

        self.conv_5 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1))
        self.m_pool_5 = nn.MaxPool2d((2, 2))

        self.fc1 = nn.Linear(14 * 14 * 128, 20)
        self.fc4 = nn.Linear(14 *14 * 128, 200)

    def forward(self, frame):


        c_1 = self.m_pool_1(F.relu(self.conv_1(frame)))
        c_2 = self.m_pool_2(F.relu(self.conv_2(c_1)))
        c_3 = self.m_pool_3(F.relu(self.conv_3(c_2)))
        c_4 = self.m_pool_4(F.relu(self.conv_4(c_3)))
        c_5 = self.m_pool_5((self.conv_5(c_4)))

        c_5_reshape = torch.transpose(c_5, 1, -1)
        c_5_flat = c_5_reshape.reshape(1, -1)

        fc1_shape_output_20 = self.fc1(c_5_flat) # 25

        fc4_pose_output_200 = F.relu(self.fc4(c_5_flat))
        return fc1_shape_output_20, fc4_pose_output_200

class ShapeOptNet(nn.Module):
    def __init__(self, BaseNet):

        super(ShapeOptNet, self).__init__()
        self.trans = torch.tensor([0., 0.2, -2.3])
        self.pose_mean = np.load(os.path.join(os.path.dirname(__file__), './../assets/mean_a_pose.npy'))
        self.pose_mean[:3] = 0.

        with open(os.path.join(os.path.dirname(__file__), './../assets/smpl_sampling.pkl'), 'rb') as f:
            sampling = pkl.load(f, encoding="latin1")

        mesh_mats = sampling['meshes']
        self.faces_coo = mesh_mats[0]['f'].astype(np.int32)
        self.up_mats = sampling['up']
        self.adj_mats = sampling['adjacency']
        self.n_model = "./assets/neutral_smpl.pkl"
        self.smpl = SMPL(self.n_model)
        self.smpl_face = SMPL(self.n_model)

        # shape_opt_layers

        self.BaseNet = BaseNet
        self.resolution_size = 862 # res_low

        out_s1 = self.resolution_size * 64
        self.fc2 = nn.Linear(20, 862 * 64)

        self.gcl_layer_3 = Gcl(32, 64)
        self.gcl_layer_2 = Gcl(16, 32)
        self.gcl_layer_1 = Gcl(16, 16)
        self.gcl_layer_0 = Gcl(3, 16)
        self.fc3 = nn.Linear(20, 10)

        #pose_opt_layers

        self.fc5 = nn.Linear(75, 200)
        self.fc6 = nn.Linear(400, 100)
        self.fc7 = nn.Linear(100, 219)

    def forward(self, segmentations, poses):

        shape_layer_20 = []
        add_poses_list = []
        add_trans_list = []
        for i in range(0, len(segmentations)):

            shape_L, pose_L_from_frames = self.BaseNet(segmentations[i])
            shape_layer_20.append(shape_L)
            pose_flat = poses[i].reshape(1, -1)

            #pose_opt_layers

            pose_layer_200_from_keypoints = F.relu(self.fc5(pose_flat))
            concat_pose_frames_keypoints = torch.cat((pose_L_from_frames, pose_layer_200_from_keypoints), dim=1)
            pose_feature_frames_keypoints = self.fc6(concat_pose_frames_keypoints)
            pose_results = self.fc7(pose_feature_frames_keypoints)

            pose_batch = batch_rodrigues(self.pose_mean.reshape(-1, 3).astype(np.float32))
            pose_init = torch.reshape(pose_batch, (-1,))
            seg_batch_size = segmentations[0].size()[0]
            pose_concat_trans = torch.cat((self.trans, pose_init), 0).unsqueeze(0).repeat(seg_batch_size, 1)
            add_poses_trans = pose_results.add(pose_concat_trans)
            add_poses_list.append(add_poses_trans[:, 3:].view(-1, 24, 3, 3))
            add_trans_list.append(add_poses_trans[:, :3])


        betas_avr = torch.mean(torch.cat(shape_layer_20),0)
        betas_val = self.fc3(betas_avr).unsqueeze(0)

        pre_gcl_linear_layer = self.fc2(betas_avr)
        reshape_pre_gcl_linear_layer = pre_gcl_linear_layer.view(self.resolution_size, 64)

        up_mats_to_tensor = [sparse_mat_to_sorted_sparse(up_mat).float() for up_mat in self.up_mats]
        adj_mat = [map(sparse_mat_to_sorted_sparse, chebyshev_polynomials(sp_mat, 3)) for sp_mat in self.adj_mats]

        gcl_l_3 = F.relu(self.gcl_layer_3(reshape_pre_gcl_linear_layer, adj_mat[3]))
        mm_adj_xw_3 = torch.sparse.mm(up_mats_to_tensor[2], gcl_l_3)

        gcl_l_2 = F.relu(self.gcl_layer_2(mm_adj_xw_3, adj_mat[2]))
        mm_adj_xw_2 = torch.sparse.mm(up_mats_to_tensor[1], gcl_l_2)

        gcl_l_1 = F.relu(self.gcl_layer_1(mm_adj_xw_2, adj_mat[1]))
        mm_adj_xw_1 = torch.sparse.mm(up_mats_to_tensor[0], gcl_l_1)

        gcl_l_0 = torch.tanh(self.gcl_layer_0(mm_adj_xw_1, adj_mat[0]))
        offsets = (gcl_l_0 / 10).unsqueeze(0)

        smpl_vertices = []

        num_f = 8

        for i in range(0, num_f):
            smpl_vertices.append(self.smpl(add_poses_list[i], betas_val, add_trans_list[i], offsets))

        smpl_vertices[0].retain_grad() # del

        smpl_v_shaped_personal= self.smpl.v_shaped_personal
        smpl_v_shaped = self.smpl.v_shaped

        lap_0 = laplacian_tensor(smpl_v_shaped_personal, self.faces_coo)
        lap_1 = laplacian_tensor(smpl_v_shaped, self.faces_coo)

        shaped_personal_mm = torch.sparse.mm(lap_0, smpl_v_shaped_personal[0])
        shaped_mm = torch.sparse.mm(lap_1, smpl_v_shaped[0])

        laplace_res = shaped_personal_mm - shaped_mm
        sym = smpl_v_shaped_personal[0]

        vert_face_torch = []
        vert_joint_torch = []
        v_personal_t = torch.zeros((1, 6890, 3)).repeat(betas_val.size()[0], 1, 1)

        for i in range(0, num_f):
            smpl_face_vert = self.smpl_face(add_poses_list[i], betas_val, add_trans_list[i], v_personal_t)
            vert_face = torch.cat((joints_body25(smpl_face_vert[0]), face_landmarks(smpl_face_vert[0])), dim=0)
            vert_face_torch.append(vert_face[25:, :])
            vert_joint_torch.append(vert_face[:25, :])

        ren_out = []

        smpl_cuda = []
        for  j in range(0, num_f):
            smpl_cuda.append(smpl_vertices[j].to(device))

        ren_o_3d = []
        for m_v in smpl_cuda:
            ren_o_3d.append(render(m_v, self.faces_coo, 1080, 0.1, 10, device))

        for  j in range(0, num_f):
            ren_out.append(ren_o_3d[j].cpu())

        return offsets, betas_val, add_poses_list, add_trans_list, self.faces_coo,\
            smpl_vertices, vert_joint_torch, vert_face_torch, laplace_res, sym, ren_out





