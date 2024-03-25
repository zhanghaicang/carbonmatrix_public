import torch
from torch.nn import functional as F
from einops import rearrange

from carbondesign.model import r3
from carbondesign.common import residue_constants
from carbondesign.model.utils import batched_select

def torsion_angles_to_frames(aatype, backb_to_global, torsion_angles_sin_cos):
    num_batch, num_residues = aatype.shape
    device = aatype.device

    # r3.Rigids with shape (batch, N, 8)
    m = batched_select(torch.tensor(residue_constants.restype_rigid_group_default_frame, device=aatype.device), aatype)
    default_frames = r3.rigids_from_tensor4x4(m)

    sin_angles = torsion_angles_sin_cos[..., 0]
    cos_angles = torsion_angles_sin_cos[..., 1]
    
    sin_angles = F.pad(sin_angles, (1, 0), value=0.) 
    cos_angles = F.pad(cos_angles, (1, 0), value=1.)
    zeros = torch.zeros_like(sin_angles)
    ones = torch.ones_like(sin_angles)

    # all_rots are r3.Rots with shape (N, 8)
    all_rots = torch.stack([
        ones, zeros, zeros,
        zeros, cos_angles, -sin_angles,
        zeros, sin_angles, cos_angles], dim=-1)
    all_rots = rearrange(all_rots, '... (h w) -> ... h w', h=3, w=3)

    all_frames = r3.rigids_mul_rots(default_frames, all_rots)

    chi2_frame_to_frame = tuple(map(lambda x: x[:, :, 5], all_frames))
    chi3_frame_to_frame = tuple(map(lambda x: x[:, :, 6], all_frames))
    chi4_frame_to_frame = tuple(map(lambda x: x[:, :, 7], all_frames))

    chi1_frame_to_backb = tuple(map(lambda x: x[:, :, 4], all_frames))
    chi2_frame_to_backb = r3.rigids_mul_rigids(chi1_frame_to_backb,
                                               chi2_frame_to_frame)
    chi3_frame_to_backb = r3.rigids_mul_rigids(chi2_frame_to_backb,
                                               chi3_frame_to_frame)
    chi4_frame_to_backb = r3.rigids_mul_rigids(chi3_frame_to_backb,
                                               chi4_frame_to_frame)

    # Recombine them to a r3.Rigids with shape (N, 8).
    def _concat_frames(xall, x5, x6, x7):
        return torch.cat([xall[:, :, 0:5], x5[:, :, None], x6[:, :, None], x7[:, :, None]], dim=2)

    all_frames_to_backb = tuple(_concat_frames(*x) for x in zip(all_frames, chi2_frame_to_backb, chi3_frame_to_backb, chi4_frame_to_backb))

    # shape (N, 8)
    all_frames_to_global = r3.rigids_mul_rigids(
            r3.rigids_op(backb_to_global,
                lambda x: x[:,:,None].repeat([1,1,8] + [1] * (x.ndim - 2))),
            all_frames_to_backb)

    return all_frames_to_global

def frames_and_literature_positions_to_atom14_pos(aatype, all_frames_to_global):
    num_batch, num_residues = aatype.shape
    device = aatype.device

    residx_to_group_idx = batched_select(
        torch.tensor(residue_constants.restype_atom14_to_rigid_group, device=device), aatype)

    # r3.Rigids with shape (N, 14)
    map_atoms_to_global = r3.rigids_op(all_frames_to_global, lambda x: batched_select(x, residx_to_group_idx, batch_dims=2))

    # r3.Vecs with shape (N, 14)
    lit_positions = batched_select(torch.tensor(residue_constants.restype_atom14_rigid_group_positions, device=device), aatype)

    # r3.Vecs with shape (N, 14)
    pred_positions = r3.rigids_mul_vecs(map_atoms_to_global, lit_positions)

    return pred_positions
