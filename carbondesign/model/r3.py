import torch
from einops import rearrange

def point_square_distance(a, b):
    return torch.sum(torch.square(a-b), dim=-1)

def point_distance(a, b):
    eps = 1e-12
    return torch.sqrt(point_square_distance(a, b) + eps)

def rigids_op(rigids, op):
    return tuple(map(op, rigids))

def rigids_mul_vecs(rigids, vecs):
    rots, trans = rigids
    assert vecs.ndim - trans.ndim  in [0, 1]

    if vecs.ndim == trans.ndim:
        #return trans + torch.einsum('... r d, ... d -> ... r', rots, vecs)
        return trans + torch.squeeze(torch.matmul(rots, vecs[..., None]), dim=-1) 
    else:
        return rearrange(trans, '... d -> ... () d') + torch.einsum('... r d, ... m d -> ... m r', rots, vecs)

def rots_mul_rots(rots_a, rots_b):
    assert rots_a.shape[-2:] == (3, 3) and rots_b.shape[-2:] == (3, 3)

    return torch.einsum('... r d, ... d m -> ... r m', rots_a, rots_b)

def rigids_mul_rots(rigids, rots_b):
    rots, trans = rigids
    return (rots_mul_rots(rots, rots_b), trans)

def rigids_mul_rigids(rigids_a, rigids_b):
    rots_a, trans_a = rigids_a
    rots_b, trans_b = rigids_b

    assert rots_a.ndim == rots_b.ndim and trans_a.ndim == trans_b.ndim

    rots = torch.einsum('... r d, ... d m -> ... r m', rots_a, rots_b)

    trans = torch.einsum('... r d, ...d -> ... r', rots_a, trans_b) + trans_a

    return (rots, trans)

def invert_rots(rots):
    return rearrange(rots, '... i j -> ... j i')

def rots_mul_vecs(rots, vecs):
    return torch.einsum('... r d, ... d -> ... r', rots, vecs)

def invert_rigids(rigids):
    rots, trans = rigids
    inv_rots = invert_rots(rots)
    inv_trans = -rots_mul_vecs(inv_rots, trans)
    
    return (inv_rots, inv_trans)


def rigids_from_tensor4x4(m):
    assert m.shape[-2:] == (4, 4)

    rots, trans = m[...,:3,:3], m[...,:3,3]

    return (rots, trans)

def vecs_robust_normalize(v, dim=-1, epsilon=1e-8):
  norms = torch.sqrt(torch.sum(torch.square(v), dim=dim, keepdims=True) + epsilon)
  return v / norms

def vecs_cross_vecs(v1, v2):
    assert v1.shape[-1] == 3 and v2.shape[-1] == 3
    
    return torch.stack([
        v1[...,1] * v2[...,2] - v1[...,2] * v2[...,1],
        v1[...,2] * v2[...,0] - v1[...,0] * v2[...,2],
        v1[...,0] * v2[...,1] - v1[...,1] * v2[...,0],
        ], dim=-1)

def rigids_from_3_points(point_on_neg_x_axis, origin, point_on_xy_plane, epsilon=1e-6):
    # Shape (b, l, 3)
    assert point_on_neg_x_axis.shape[-1] == 3\
           and origin.shape[-1] == 3\
           and point_on_xy_plane.shape[-1] == 3

    e0_unnormalized = origin - point_on_neg_x_axis

    e1_unnormalized = point_on_xy_plane - origin

    e0 = vecs_robust_normalize(e0_unnormalized)
    
    c = torch.einsum('... c, ... c -> ... ', e1_unnormalized, e0)[..., None]
    e1 = e1_unnormalized - c * e0
    e1 = vecs_robust_normalize(e1)

    # e2 = torch.cross(e0, e1, dim=-1)
    e2 = vecs_cross_vecs(e0, e1)

    R = torch.stack((e0, e1, e2), dim=-1)

    return (R, origin)
