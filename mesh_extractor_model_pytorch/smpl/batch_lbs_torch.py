import torch

def batch_skew(vec, batch_size=None):
    """
    vec is N x 3, batch_size is int

    returns N x 3 x 3. Skew_sym version of each matrix.
    """
    if batch_size is None:
         batch_size = vec.size()[0]
    col_inds = torch.tensor([1, 2, 3, 5, 6, 7], dtype=torch.int)

    indices = torch.reshape(torch.reshape(torch.arange(0, batch_size, step=1) * 9, (-1, 1)) + col_inds,(-1, 1))

    updates = torch.reshape(
    torch.stack((
                    -vec[:, 2], vec[:, 1], vec[:, 2], -vec[:, 0], -vec[:, 1],
                    vec[:, 0]
                ), dim=1), (-1,))

    out_shape = (batch_size * 9,)
    res = torch.zeros(out_shape, dtype = torch.float32)
    res[indices[:, 0]] = updates
    res = torch.reshape(res, (batch_size, 3, 3))

    return res

def batch_rodrigues(theta, name=None):
    """
    Theta is N x 3
    """
    theta_t = torch.from_numpy(theta)
    batch_size = theta_t.size()[0]
    angle = torch.norm(theta_t + 1e-8, dim=1).unsqueeze(-1)
    r = torch.div(theta_t, angle).unsqueeze(-1)

    angle = angle.unsqueeze(-1)
    cos = torch.cos(angle)
    sin = torch.sin(angle)

    outer = torch.matmul(r, r.permute(0, 2, 1))
    eyes = torch.eye(3).unsqueeze(0).repeat([batch_size, 1, 1])

    R = cos * eyes + (1 - cos) * outer + sin * batch_skew(
            r, batch_size=batch_size)
    return R
