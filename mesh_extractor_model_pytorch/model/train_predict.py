import torch
import torch.nn as nn
import numpy as np
from lib.lib_fun import reproj_fun
from smpl.bodyparts import regularize_laplace, regularize_symmetry
import os
import pickle as pkl
from tqdm import tqdm

def laplace_mse(ypred):
    w = regularize_laplace()
    return torch.mean(torch.tensor(w[np.newaxis, :, np.newaxis]) * torch.square(ypred), -1)

def symmetry_mse(ypred):
    w = regularize_symmetry()
    idx = np.load(os.path.join(os.path.dirname(__file__), './../assets/vert_sym_idxs.npy'))
    ypred_mirror = ypred[:, idx, :] * torch.tensor([-1., 1., 1.], dtype=torch.float32).reshape(1, 1, 3)
    return torch.mean(torch.tensor(w[np.newaxis, :, np.newaxis]) * torch.square(ypred - ypred_mirror), -1)


def fine_tuning(model, base_model, weights_dir, input_seg, input_pose,input_face, num_f, pose_step, shape_step, out_dir):

    ## Load weights
    model.load_state_dict(torch.load(weights_dir))

    ## Model parameter status
    base_model.conv_1.weight.requires_grad = False
    base_model.conv_1.bias.requires_grad = False
    base_model.conv_2.weight.requires_grad = False
    base_model.conv_2.bias.requires_grad = False
    base_model.conv_3.weight.requires_grad = False
    base_model.conv_3.bias.requires_grad = False
    base_model.conv_4.weight.requires_grad = False
    base_model.conv_4.bias.requires_grad = False
    base_model.conv_5.weight.requires_grad = False
    base_model.conv_5.bias.requires_grad = False

    base_model.fc1.weight.requires_grad = True
    base_model.fc1.bias.requires_grad = True

    base_model.fc4.weight.requires_grad = False
    base_model.fc4.bias.requires_grad = False

    model.gcl_layer_3.weight_block_0.requires_grad = False
    model.gcl_layer_3.weight_block_1.requires_grad = False
    model.gcl_layer_3.weight_block_2.requires_grad = False
    model.gcl_layer_3.weight_block_3.requires_grad = False
    model.gcl_layer_3.bias.requires_grad = False

    model.gcl_layer_2.weight_block_0.requires_grad = False
    model.gcl_layer_2.weight_block_1.requires_grad = False
    model.gcl_layer_2.weight_block_2.requires_grad = False
    model.gcl_layer_2.weight_block_3.requires_grad = False
    model.gcl_layer_2.bias.requires_grad = False

    model.gcl_layer_1.weight_block_0.requires_grad = False
    model.gcl_layer_1.weight_block_1.requires_grad = False
    model.gcl_layer_1.weight_block_2.requires_grad = False
    model.gcl_layer_1.weight_block_3.requires_grad = False
    model.gcl_layer_1.bias.requires_grad = False

    model.gcl_layer_0.weight_block_0.requires_grad = True
    model.gcl_layer_0.weight_block_1.requires_grad = True
    model.gcl_layer_0.weight_block_2.requires_grad = True
    model.gcl_layer_0.weight_block_3.requires_grad = True
    model.gcl_layer_0.bias.requires_grad = True

    model.fc3.weight.requires_grad = False
    model.fc3.bias.requires_grad = False

    model.fc5.weight.requires_grad = False
    model.fc5.bias.requires_grad = False

    model.fc2.weight.requires_grad = True
    model.fc2.bias.requires_grad = True

    model.fc6.weight.requires_grad = True
    model.fc6.bias.requires_grad = True

    model.fc7.weight.requires_grad = False
    model.fc7.bias.requires_grad = False

    model.train()

    ## optimization
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-07, weight_decay=0.0, amsgrad=False)

    render_loss = nn.MSELoss(reduction="mean")
    loss_s = []
    loss_p = []
    img_size = 1080
    repr_loss_torch = reproj_fun([img_size, img_size],
                       [img_size / 2., img_size / 2.],
                                  img_size, img_size)

    input_ren = []
    for i in range(0, num_f):
        im_in_ren = torch.transpose(input_seg[i], 1, -1)
        input_segmentation_r = im_in_ren[0, :, :, 0]
        input_segmentation_g = im_in_ren[0, :, :, 1]
        input_segmentation_b = im_in_ren[0, :, :, 2]
        input_segmentation = input_segmentation_r + input_segmentation_g + input_segmentation_b
        input_segmentation[input_segmentation > 0] =  1
        input_ren.append(input_segmentation)


    for k in tqdm(range(0, pose_step + shape_step), ascii=True, desc="optimization"):
        optimizer.zero_grad()
        offsets, betas, pose_list, trans_list, faces_coo, smpl_vertices,\
        vert_joint_torch,vert_face_torch, laplace, sym, ren_o = model(input_seg, input_pose)
        pose_criterion = repr_loss_torch(input_pose[0].unsqueeze(0), vert_joint_torch[0].unsqueeze(0))

        for i in range(1, num_f):
            pose_criterion = pose_criterion.add(repr_loss_torch(input_pose[i].unsqueeze(0),vert_joint_torch[i].unsqueeze(0)))

        if k < pose_step:
            optimizer.zero_grad()
            pose_loss = pose_criterion
            pose_loss.backward(retain_graph=True)
            optimizer.step()

        if k >= pose_step and k < (shape_step + pose_step):
            shape_loss_1 = torch.sum(laplace_mse(laplace[None, :, :]))/6890
            shape_loss_2 = torch.sum(symmetry_mse(sym[None, :, :]))/6890
            shape_loss_3 = repr_loss_torch(input_pose[0].unsqueeze(0), vert_joint_torch[0].unsqueeze(0))
            shape_loss_4 = repr_loss_torch(input_face[0].unsqueeze(0), vert_face_torch[0].unsqueeze(0))
            shape_loss_5 = render_loss(ren_o[0], input_ren[0])

            for i in range(1, num_f):
                loss_J = repr_loss_torch(input_pose[i].unsqueeze(0), vert_joint_torch[i].unsqueeze(0))
                shape_loss_3 = shape_loss_3.add(loss_J)
                loss_F = repr_loss_torch(input_face[i].unsqueeze(0), vert_face_torch[i].unsqueeze(0))
                shape_loss_4 = shape_loss_4.add(loss_F)
                loss_R = render_loss(ren_o[i], input_ren[i])
                shape_loss_5 = shape_loss_5.add(loss_R)

            shape_loss = 1 * ( 100 * num_f * shape_loss_1 + 50 * num_f * shape_loss_2 + 50 * shape_loss_3 + \
            10 * num_f * shape_loss_4) + 1 * shape_loss_5
            loss_s.append(shape_loss)
            shape_loss.backward()
            optimizer.step()

    ## prediction
    model.eval()
    offsets, betas, pose_list, trans_list, faces_coo_p, smpl_vertices_p,\
    vert_joint_torch,vert_face_torch, laplace, sym, ren_o = model(input_seg, input_pose)
    def save_mesh(filename, vv, f, vt, ft):
        with open(filename, 'w') as fp:
            v = vv[0]
            fp.write(('v {:f} {:f} {:f}\n' * len(v)).format(*v.reshape(-1)))
            fp.write(('f {:d}/{:d}/{:d} {:d}/{:d}/{:d} {:d}/{:d}/{:d}\n' * len(f)).format(*np.hstack((f.reshape(-1, 1),\
            ft.reshape(-1, 1), f.reshape(-1, 1))).reshape(-1) + 1))
            fp.write(('vt {:f} {:f}\n' * len(vt)).format(*vt.reshape(-1)))

    vt = np.load('assets/basicModel_vt.npy')
    ft = np.load('assets/basicModel_ft.npy')

    save_mesh(out_dir + '/mesh_object.obj', smpl_vertices_p[0], faces_coo_p, vt, ft)

    width = 1080
    height = 1080
    frame_data = {
        'width': width,
        'camera_c': np.array([width, height]) / 2.,
        'vertices': np.zeros((num_f, 6890, 3)),
        'camera_f': np.array([width, width]),
        'height': height,
    }

    pred_vert = np.zeros((num_f, 6890, 3))
    for i in range(0, num_f):
        pred_vert[i, :, :] = smpl_vertices_p[i].detach().numpy()

    frame_data['vertices'] = pred_vert
    with open(out_dir + "/frame_data.pkl", 'wb') as f:
        pkl.dump(frame_data, f, 2)

    print('Done.')
