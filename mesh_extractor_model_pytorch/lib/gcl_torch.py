import sys
import os

import torch.nn as nn
import torch.nn.functional as F
import torch

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import pickle as pkl

# https://github.com/tkipf/gcn/blob/master/gcn/utils.py#L149
def chebyshev_polynomials(adj, k):

    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0], dtype=adj.dtype) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0], dtype=adj.dtype)

    t_k = list()
    t_k.append(sp.eye(adj.shape[0], dtype=adj.dtype))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True, dtype=adj.dtype)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k + 1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return t_k


# https://github.com/tkipf/gcn/blob/master/gcn/utils.py#L122
def normalize_adj(adj):

    """Symmetrically normalize adjacency matrix."""

    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def preprocess_adj(adj):
    adj = (adj > 0).astype(adj.dtype)
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0], dtype=adj.dtype))
    return adj_normalized

def sparse_mat_to_sorted_sparse(x):

    coo = x.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    ind_i = coo.row
    ind_j = coo.col
    val_sparse = coo.data

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

    s_torch = torch.sparse.FloatTensor(i, v, torch.Size(coo.shape))
    return s_torch

class Gcl(nn.Module):
    def __init__(self, input_size, output_size):
        super(Gcl, self).__init__()

        self.input_dim = input_size
        self.output_dim = output_size

        self.weight_block_0 = torch.nn.Parameter(data=torch.Tensor(self.output_dim, self.input_dim))
        self.weight_block_1 = torch.nn.Parameter(data=torch.Tensor(self.output_dim, self.input_dim))
        self.weight_block_2 = torch.nn.Parameter(data=torch.Tensor(self.output_dim, self.input_dim))
        self.weight_block_3 = torch.nn.Parameter(data=torch.Tensor(self.output_dim, self.input_dim))
        self.bias = torch.nn.Parameter(data=torch.Tensor(self.input_dim))

        self.weight_block_0.data.uniform_(-1, 1)
        self.weight_block_1.data.uniform_(-1, 1)
        self.weight_block_2.data.uniform_(-1, 1)
        self.weight_block_3.data.uniform_(-1, 1)
        self.bias.data.uniform_(-1, 1)

    def forward(self, x, adj_mat):

        weights_block_list = [self.weight_block_0, self.weight_block_1, self.weight_block_2, self.weight_block_3]
        sparse_adj_list = []

        for adj, w in zip(adj_mat, weights_block_list):

            xw = torch.matmul(x, w).type(torch.FloatTensor)
            mm_adj_xw = torch.sparse.mm(adj, xw)
            sparse_adj_list.append(mm_adj_xw)

        gcl_out = sparse_adj_list[0]
        for i in range(1, len(sparse_adj_list)):
            gcl_out = gcl_out + sparse_adj_list[i]

        return gcl_out + self.bias


if __name__ == "__main__":


    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.gcl_1 = Gcl(32, 64)

        def forward(self,x, adj_mat):
            x = F.relu(self.gcl_1(x, adj_mat))
            return x


    with open(os.path.join(os.path.dirname(__file__), './../../assets/smpl_sampling.pkl'), 'rb') as f:
        sampling = pkl.load(f, encoding="latin1")

    M = sampling['meshes']
    U = sampling['up']
    D = sampling['down']
    A = sampling['adjacency']

    adj_mat = [map(sparse_mat_to_sorted_sparse, chebyshev_polynomials(a, 3)) for a in A]

    model = Model()

    model.gcl_1.weight_block_0.data = torch.from_numpy(np.ones((64, 32)))
    model.gcl_1.weight_block_1.data = torch.from_numpy(np.ones((64, 32)))
    model.gcl_1.weight_block_2.data = torch.from_numpy(np.ones((64, 32)))
    model.gcl_1.weight_block_3.data = torch.from_numpy(np.ones((64, 32)))
    model.gcl_1.bias.data =  torch.from_numpy(2*np.ones(32))

    x = 5*np.ones(862 * 64).reshape(1, 1, -1).reshape(862 , 64)
    out = model(x, adj_mat[3])

    print("output", out)
    print(out.shape)
