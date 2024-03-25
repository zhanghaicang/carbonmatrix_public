import numpy as np
import torch
from torch.nn import functional as F

from einops import rearrange

from carbondesign.model.utils import l2_normalize

# pylint: disable=bad-whitespace
QUAT_TO_ROT = np.zeros((4, 4, 3, 3), dtype=np.float32)

QUAT_TO_ROT[0, 0] = [[ 1, 0, 0], [ 0, 1, 0], [ 0, 0, 1]]  # rr
QUAT_TO_ROT[1, 1] = [[ 1, 0, 0], [ 0,-1, 0], [ 0, 0,-1]]  # ii
QUAT_TO_ROT[2, 2] = [[-1, 0, 0], [ 0, 1, 0], [ 0, 0,-1]]  # jj
QUAT_TO_ROT[3, 3] = [[-1, 0, 0], [ 0,-1, 0], [ 0, 0, 1]]  # kk

QUAT_TO_ROT[1, 2] = [[ 0, 2, 0], [ 2, 0, 0], [ 0, 0, 0]]  # ij
QUAT_TO_ROT[1, 3] = [[ 0, 0, 2], [ 0, 0, 0], [ 2, 0, 0]]  # ik
QUAT_TO_ROT[2, 3] = [[ 0, 0, 0], [ 0, 0, 2], [ 0, 2, 0]]  # jk

QUAT_TO_ROT[0, 1] = [[ 0, 0, 0], [ 0, 0,-2], [ 0, 2, 0]]  # ir
QUAT_TO_ROT[0, 2] = [[ 0, 0, 2], [ 0, 0, 0], [-2, 0, 0]]  # jr
QUAT_TO_ROT[0, 3] = [[ 0,-2, 0], [ 2, 0, 0], [ 0, 0, 0]]  # kr

QUAT_MULTIPLY = np.zeros((4, 4, 4), dtype=np.float32)
QUAT_MULTIPLY[:, :, 0] = [[ 1, 0, 0, 0],
                          [ 0,-1, 0, 0],
                          [ 0, 0,-1, 0],
                          [ 0, 0, 0,-1]]

QUAT_MULTIPLY[:, :, 1] = [[ 0, 1, 0, 0],
                          [ 1, 0, 0, 0],
                          [ 0, 0, 0, 1],
                          [ 0, 0,-1, 0]]

QUAT_MULTIPLY[:, :, 2] = [[ 0, 0, 1, 0],
                          [ 0, 0, 0,-1],
                          [ 1, 0, 0, 0],
                          [ 0, 1, 0, 0]]

QUAT_MULTIPLY[:, :, 3] = [[ 0, 0, 0, 1],
                          [ 0, 0, 1, 0],
                          [ 0,-1, 0, 0],
                          [ 1, 0, 0, 0]]

QUAT_MULTIPLY_BY_VEC = QUAT_MULTIPLY[:, 1:, :]
# pylint: enable=bad-whitespace

QUAT_TO_ROT_TORCH = torch.tensor(np.reshape(QUAT_TO_ROT, (4, 4, 9)))
QUAT_MULTIPLY_TORCH = torch.tensor(QUAT_MULTIPLY)
QUAT_MULTIPLY_BY_VEC_TORCH = torch.tensor(QUAT_MULTIPLY_BY_VEC)

def make_identity(out_shape, device):
    out_shape = (out_shape) + (3,)
    quaternions = F.pad(torch.zeros(out_shape, device=device), (1, 0), value=1.)
    translations = torch.zeros(out_shape, device = device)

    return quaternions, translations

def quat_to_rot(normalized_quat):
    rot_tensor = torch.sum(
            QUAT_TO_ROT_TORCH.to(normalized_quat.device) *
            normalized_quat[..., :, None, None] *
            normalized_quat[..., None, :, None],
            axis=(-3, -2))
    rot = rearrange(rot_tensor, '... (c d) -> ... c d', c=3, d=3)
    return rot

def quat_multiply_by_vec(quat, vec):
    return torch.sum(
            QUAT_MULTIPLY_BY_VEC_TORCH.to(quat.device) *
            quat[..., :, None, None] *
            vec[..., None, :, None],
            dim=(-3, -2))

def quat_multiply(quat1, quat2):
    assert quat1.shape == quat2.shape
    return torch.sum(
            QUAT_MULTIPLY_TORCH.to(quat1.device) *
            quat1[..., :, None, None] *
            quat2[..., None, :, None],
            dim=(-3, -2))

def quat_precompose_vec(quaternion, vector_quaternion_update):
    assert quaternion.shape[-1] == 4\
            and vector_quaternion_update.shape[-1] == 3\
            and quaternion.shape[:-1] == vector_quaternion_update.shape[:-1]
            
    new_quaternion = quaternion + quat_multiply_by_vec(quaternion, vector_quaternion_update)
    normalized_quaternion = l2_normalize(new_quaternion)

    return normalized_quaternion
