from __future__ import division
import numpy as np
import torch
import cv2
import json

def to_sparse_coo(ind_i, ind_j, val_sparse, sparse_shape):

    sort_ind = np.argsort(val_sparse)[::-1]

    sorted_val_sparse = list()
    sorted_ind_i = list()
    sorted_ind_j = list()
    for ind in sort_ind:

        sorted_val_sparse.append(val_sparse[ind])
        sorted_ind_i.append(ind_i[ind])
        sorted_ind_j.append(ind_j[ind])

    i = torch.LongTensor([sorted_ind_i, sorted_ind_j])
    v = torch.FloatTensor(sorted_val_sparse)

    s_torch = torch.sparse.FloatTensor(i, v, torch.Size(sparse_shape))
    return s_torch

def prepare_segmentations(file):

    LABELS_FULL = {
    'Sunglasses': [170, 0, 51],
    'LeftArm': [51, 170, 221],
    'RightArm': [0, 255, 255],
    'LeftLeg': [85, 255, 170],
    'RightLeg': [170, 255, 85],
    'LeftShoe': [255, 255, 0],
    'RightShoe': [255, 170, 0],
    }

    LABELS_CLOTHING= {
    'Face': [0, 0, 255],
    'Arms': [51, 170, 221],
    'Legs': [85, 255, 170],
    'Shoes': [255, 255, 0]
     }

    segm = cv2.imread(file)[:, :, ::-1]

    segm[np.all(segm == LABELS_FULL['Sunglasses'], axis=2)] = LABELS_CLOTHING['Face']
    segm[np.all(segm == LABELS_FULL['LeftArm'], axis=2)] = LABELS_CLOTHING['Arms']
    segm[np.all(segm == LABELS_FULL['RightArm'], axis=2)] = LABELS_CLOTHING['Arms']
    segm[np.all(segm == LABELS_FULL['LeftLeg'], axis=2)] = LABELS_CLOTHING['Legs']
    segm[np.all(segm == LABELS_FULL['RightLeg'], axis=2)] = LABELS_CLOTHING['Legs']
    segm[np.all(segm == LABELS_FULL['LeftShoe'], axis=2)] = LABELS_CLOTHING['Shoes']
    segm[np.all(segm == LABELS_FULL['RightShoe'], axis=2)] = LABELS_CLOTHING['Shoes']

    segm_out = np.float32(np.transpose(segm[:, :, ::-1] / 255.))
    return torch.tensor(segm_out, dtype=torch.float32)

def openpose_from_file(file, resolution=(1080, 1080), person=0):
    with open(file) as f:
        data = json.load(f)['people'][person]

        pose = np.array(data['pose_keypoints_2d']).reshape(-1, 3)
        pose[:, 2] /= np.expand_dims(np.mean(pose[:, 2][pose[:, 2] > 0.1]), -1)
        pose = pose * np.array([2. / resolution[1], -2. / resolution[0], 1.]) + np.array([-1., 1., 0.])
        pose[:, 0] *= 1. * resolution[1] / resolution[0]

        face = np.array(data['face_keypoints_2d']).reshape(-1, 3)
        face = face * np.array([2. / resolution[1], -2. / resolution[0], 1.]) + np.array([-1., 1., 0.])
        face[:, 0] *= 1. * resolution[1] / resolution[0]

        return pose, face

def laplacian_tensor(vert, face):

    vert_size_0 = vert.size()[0]
    vert_size_1 = vert.size()[1]
    face_size = face.shape[0]

    vert_1 = face[:, 0]
    vert_2 = face[:, 1]
    vert_3 = face[:, 2]

    vert_1_t = torch.tensor(vert_1)
    vert_2_t = torch.tensor(vert_2)
    vert_3_t = torch.tensor(vert_3)

    ind_1 = vert[:, vert_1]
    ind_2 = vert[:, vert_2]
    ind_3 = vert[:, vert_3]

    ind_12 = ind_1 - ind_2
    ind_23 = ind_2 - ind_3
    ind_31 = ind_3 - ind_1

    red_ind_1 =-1 * torch.sum(ind_12 * ind_31, dim=2) / torch.sqrt(torch.sum(torch.cross(ind_12, ind_31, dim=-1) ** 2, dim=-1))
    red_ind_2 =-1 * torch.sum(ind_23 * ind_12, dim=2) / torch.sqrt(torch.sum(torch.cross(ind_23, ind_12, dim=-1) ** 2, dim=-1))
    red_ind_3 =-1 * torch.sum(ind_31 * ind_23, dim=2) / torch.sqrt(torch.sum(torch.cross(ind_31, ind_23, dim=-1) ** 2, dim=-1))

    tile_i = (torch.cat((vert_1_t, vert_3_t, vert_1_t, vert_2_t, vert_2_t, vert_3_t), 0)).unsqueeze(0).repeat(vert_size_0,1)
    tile_j = (torch.cat((vert_3_t, vert_1_t, vert_2_t, vert_1_t, vert_3_t, vert_2_t), 0)).unsqueeze(0).repeat(vert_size_0, 1)

    cat_red_ind = 0.5 * torch.cat((red_ind_2, red_ind_2, red_ind_3, red_ind_3, red_ind_1, red_ind_1), dim=1).view(vert_size_0, 6, -1)

    batch_size = torch.arange(vert_size_0).unsqueeze(1).repeat(1, face_size * 6)
    ind_stack = torch.stack((batch_size.long(), tile_j.long(), tile_i.long()), dim=2).view(vert_size_0, 6, -1, 3)

    cast_ind_stack = [ind_stack[:, i].view(-1, 3).type(torch.int64) for i in range(6)]
    vert_size_stack = torch.tensor([vert_size_1, vert_size_1]).type(torch.int64)
    to_sp_mat = []

    for i in range(6):

        val_sparse = cat_red_ind[:, i].view((-1,))
        ind_i = cast_ind_stack[i][:, 1]
        ind_j = cast_ind_stack[i][:, 2]

        i = torch.LongTensor(torch.stack((ind_i, ind_j)))
        v = torch.FloatTensor(val_sparse)

        s_torch = torch.sparse.FloatTensor(i, v, torch.Size(vert_size_stack))

        to_sp_mat.append(s_torch)

    add_sp = to_sp_mat[0]
    for i in range(1, 6):
        add_sp = add_sp.add(to_sp_mat[i])

    add_sparse = torch.sparse.FloatTensor.coalesce(add_sp)

    sum_add_sparse = torch.sparse.sum(add_sparse, dim=-1) * -1

    ind_tile_vert_size = torch.arange(vert_size_1).unsqueeze(0).repeat(vert_size_0, 1)
    dim_tile_vert_size = torch.arange(vert_size_0).unsqueeze(0).repeat(1, vert_size_1)
    inds_vert_size = torch.stack((dim_tile_vert_size, ind_tile_vert_size, ind_tile_vert_size), dim=2).view(-1, 3)

    val_sparse_v = sum_add_sparse.to_dense().view((-1,))
    ind_i_v = inds_vert_size[:, 1]
    ind_j_v = inds_vert_size[:, 2]
    i_v = torch.LongTensor(torch.stack((ind_i_v, ind_j_v)))
    v_v = torch.FloatTensor(val_sparse_v)

    sparse_vert_size = torch.sparse.FloatTensor(i_v, v_v, torch.Size(vert_size_stack))

    add_sp_v = add_sp.add(sparse_vert_size)
    add_sparse_v = torch.sparse.FloatTensor.coalesce(add_sp_v)

    return add_sparse_v

def reproj_mat(foc, cent, width, height, near, far):

    foc = 0.5 * (foc[0] + foc[1])
    pix_off = 0.5
    right = (width - (cent[0] + pix_off)) * (near / foc)
    left = -(cent[0] + pix_off) * (near / foc)
    top = (cent[1] + pix_off) * (near / foc)
    bottom = -(height - cent[1] + pix_off) * (near / foc)

    proj_mat = torch.tensor([
        [2. * near / (right - left), 0., (right + left) / (right - left), 0.],
        [0., 2. * near / (top - bottom), (top + bottom) / (top - bottom), 0.],
        [0., 0., -(far + near) / (far - near), -2. * far * near / (far - near)],
        [0., 0., -1., 0.]
    ], dtype=torch.float32)

    return torch.transpose(proj_mat, 0, 1)


def reproj_fun(focal, center, width_in, height_in):

    def _reproj_loss(y_org, y_est):

        y_est_size = y_est.shape[0]
        p_mat = reproj_mat(focal, center, width_in, height_in, .1, 10)
        p_mat = p_mat.unsqueeze(0).repeat(y_est_size, 1, 1)
        y_est_h = torch.cat([y_est, torch.ones_like(y_est[:, :, -1:])], axis=2)
        y_est_p = torch.matmul(y_est_h, p_mat)
        y_est_p_d = y_est_p / y_est_p[:, :, -1].unsqueeze(-1)
        diff_s = (y_org[:, :, :2] - y_est_p_d[:, :, :2])[0]
        y_org_temp = y_org[:, :, 2]
        out_f = torch.mean(torch.square(torch.transpose(diff_s, 0, 1) * y_org_temp))

        return(out_f)

    return _reproj_loss

