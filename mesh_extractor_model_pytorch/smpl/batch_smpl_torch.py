from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import torch.nn.functional as F
import torch
#from .batch_lbs import batch_rodrigues, batch_global_rigid_transformation
import pickle as pkl

####################################
def undo_chumpy(x):
    return x if isinstance(x, np.ndarray) else x.r

#####################################
def sparse_to_tensor(x, dtype=torch.float32):
    coo = x.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return torch.sparse.FloatTensor(torch.LongTensor(indices).t(), torch.tensor(coo.data, dtype=dtype), torch.Size(coo.shape))

######################################
def batch_global_rigid_transformation(Rs, Js_, parent, rotate_base=False):

    N = Rs.size()[0]
    root_rotation = Rs[:, 0, :, :]
    Js = Js_.unsqueeze(-1)
    def make_A(R, t, name=None):

        R_homo = F.pad(input=R, pad=(0,0,0,1,0, 0), mode='constant', value=0)
        t_homo = torch.cat([t, torch.ones([N, 1, 1], dtype=torch.float32)], 1)
        return torch.cat([R_homo, t_homo], 2)

    A0 = make_A(root_rotation, Js[:, 0])
    results = [A0]
    for i in range(1, parent.shape[0]):

        j_here = Js[:, i] - Js[:, parent[i]]
        A_here = make_A(Rs[:, i], j_here)
        res_here = torch.matmul(results[parent[i]], A_here)
        results.append(res_here)
    results_ = torch.stack(results, dim=1)
    new_J = results_[:, :, :3, 3]
    Js_w0 = torch.cat([Js, torch.zeros([N, 24, 1, 1], dtype=torch.float32)], 2)
    init_bone = torch.matmul(results_.clone(), Js_w0.clone())
    init_bone_ = F.pad(input=init_bone, pad=(3,0,0,0,0,0,0,0), mode='constant', value=0)
    A = results_ - init_bone_

    return new_J, A

######################################
class SMPL(object):
    def __init__(self, pkl_path, dtype=torch.float32):

        with open(pkl_path, 'rb') as f:
            dd = pkl.load(f, encoding="latin1")

        self.v_template = torch.tensor(undo_chumpy(dd['v_template']),  dtype=dtype)
        self.size = [self.v_template.shape[0], 3]
        self.num_betas = dd['shapedirs'].shape[-1]
        shapedir = np.reshape(undo_chumpy(dd['shapedirs']), [-1, self.num_betas]).T
        shapedir_t = np.copy(shapedir)
        self.shapedirs = torch.tensor(shapedir_t, dtype=dtype)
        self.J_regressor = sparse_to_tensor(dd['J_regressor'], dtype=dtype)
        num_pose_basis = dd['posedirs'].shape[-1]
        posedirs = np.reshape(undo_chumpy(dd['posedirs']), [-1, num_pose_basis]).T
        self.posedirs = torch.tensor(posedirs, dtype=dtype)
        self.parents = dd['kintree_table'][0].astype(np.int32)
        self.weights = torch.tensor(undo_chumpy(dd['weights']), dtype=dtype)

    def __call__(self, theta, beta, trans, v_personal, name=None):

        num_batch = beta.size()[0]
        v_shaped_scaled = torch.reshape(torch.mm(beta, self.shapedirs),
                [-1, self.size[0], self.size[1]]) + self.v_template

        body_height = (v_shaped_scaled[:, 2802, 1] + v_shaped_scaled[:, 6262, 1]) - (v_shaped_scaled[:, 2237, 1] + v_shaped_scaled[:, 6728, 1])
        scale = torch.reshape(1.66 / body_height, (-1, 1, 1)).float()
        self.v_shaped = scale * v_shaped_scaled
        self.v_shaped_personal = self.v_shaped + v_personal

        Jx = torch.sparse.mm(self.J_regressor, v_shaped_scaled[:, :, 0].t()).t()
        Jy = torch.sparse.mm(self.J_regressor, v_shaped_scaled[:, :, 1].t()).t()
        Jz = torch.sparse.mm(self.J_regressor, v_shaped_scaled[:, :, 2].t()).t()
        J = scale * torch.stack([Jx, Jy, Jz], dim=2)

        u, s, v = torch.svd(theta)
        Rs = torch.matmul(u, v.permute((0, 1, 3, 2)))
        pose_feature = torch.reshape(Rs[:, 1:, :, :] - torch.eye(3), [-1, 207])

        self.v_posed = torch.reshape(
                torch.matmul(pose_feature, self.posedirs),
                [-1, self.size[0], self.size[1]]) + self.v_shaped_personal

        self.J_transformed, A_ = batch_global_rigid_transformation(Rs, J, self.parents)
        self.J_transformed = self.J_transformed + trans.unsqueeze(1)
        W = torch.reshape(self.weights.repeat(num_batch, 1), (num_batch, -1, 24))
        T = torch.reshape(torch.matmul(W, torch.reshape(A_, (num_batch, 24, 16))),(num_batch, -1, 4, 4))
        v_posed_homo = torch.cat([self.v_posed, torch.ones([num_batch, self.v_posed.size()[1], 1], dtype=torch.float32)], 2)
        v_homo = torch.matmul(T, v_posed_homo.unsqueeze(-1))
        verts = v_homo[:, :, :3, 0]
        verts_t = verts + trans.unsqueeze(1)

        return verts_t





if __name__=="__main__":


    n_model = "./assets/neutral_smpl.pkl"
    smpl = SMPL(n_model)
    offsets_np = torch.from_numpy(np.float32(np.random.normal(size=(1, 6890, 3))))
    betas_np = torch.from_numpy(np.float32(np.random.normal(size=(1, 10))))
    pose_list_np = torch.from_numpy(np.float32(np.random.normal(size=(1, 24, 3, 3))))
    trans_list_np = torch.from_numpy(np.float32(np.random.normal(size=(1, 3))))

    vertices = smpl(pose_list_np, betas_np, trans_list_np, offsets_np)

    print("-----", vertices)
    print("---A.shape", vertices.size())



    import numpy as np
    from batch_smpl import SMPL
    import keras.backend as K
    from silence_tensorflow import silence_tensorflow
    silence_tensorflow()

    smpl = SMPL(n_model, theta_in_rodrigues=False, theta_is_perfect_rotmtx=False)

    vertices = smpl(pose_list_np, betas_np, trans_list_np, offsets_np)

    print(K.get_value(vertices))
    print(vertices)


