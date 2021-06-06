
import os
import sys
import torch
import numpy as np
import pickle as pkl
from .batch_smpl_torch import sparse_to_tensor

body_25_reg = None
face_reg = None




def joints_body25(v):
    global body_25_reg

    if body_25_reg is None:

        body_25_reg = sparse_to_tensor(
            pkl.load(open(os.path.join(os.path.dirname(__file__), './../assets/J_regressor.pkl'), 'rb'), encoding='latin1').T
        )

    return torch.sparse.mm(body_25_reg.float(), v)


def face_landmarks(v):
    global face_reg

    if face_reg is None:
        face_reg = sparse_to_tensor(
            pkl.load(open(os.path.join(os.path.dirname(__file__), './../assets/face_regressor.pkl'), 'rb'), encoding='latin1').T
        )

    return torch.sparse.mm(face_reg.float(), v)
