import numpy as np
import torch
from torch.nn import functional as F

from carbondesign.model import r3
from carbondesign.common import residue_constants
from carbondesign.model.utils import batched_select

def calpha3_to_frames(calpha_pos, calpha_mask=None):
  device = calpha_pos.device
  
  # calpha_pos, (b, l, 3)
  # calpha_mask, (b, l)
  prev_calpha_pos = F.pad(calpha_pos[:, :-1], [0, 0, 1, 0])
  prev2_calpha_pos = F.pad(calpha_pos[:, :-2], [0, 0, 2, 0])
  
  next_calpha_pos = F.pad(calpha_pos[:, 1:], [0, 0, 0, 1])
  next2_calpha_pos = F.pad(calpha_pos[:, 2:], [0, 0, 0, 2])

  # (b, l, 3x3), (b, l, 3)
  left_gt_frames = r3.rigids_from_3_points(
          point_on_neg_x_axis=prev_calpha_pos,
          origin=calpha_pos,
          point_on_xy_plane=prev2_calpha_pos)

  left_forth_atom_rel_pos = r3.rigids_mul_vecs(
          r3.invert_rigids(left_gt_frames),
          next_calpha_pos)
  
  right_gt_frames = r3.rigids_from_3_points(
          point_on_neg_x_axis=next_calpha_pos,
          origin=calpha_pos,
          point_on_xy_plane=next2_calpha_pos)

  right_forth_atom_rel_pos = r3.rigids_mul_vecs(
          r3.invert_rigids(right_gt_frames),
          prev_calpha_pos)

  ret = {
      'left_gt_calpha3_frame_positions': left_forth_atom_rel_pos,
      'right_gt_calpha3_frame_positions': right_forth_atom_rel_pos,
      }
  
  if calpha_mask is not None:
      prev_calpha_mask = F.pad(calpha_mask[:, :-1], [1, 0])
      prev2_calpha_mask = F.pad(calpha_mask[:, :-2], [2, 0])
      next_calpha_mask = F.pad(calpha_mask[:, 1:], [0, 1])
      next2_calpha_mask = F.pad(calpha_mask[:, 2:], [0, 2])
      
      ret.update(
              left_gt_calpha3_frame_position_exists = torch.all(
                  torch.stack([prev2_calpha_mask, prev_calpha_mask, calpha_mask, next_calpha_mask], dim=-1), dim=-1),
              right_gt_calpha3_frame_position_exists = torch.all(
                  torch.stack([prev_calpha_mask, calpha_mask, next_calpha_mask, next2_calpha_mask], dim=-1), dim=-1)
              )
      ret['right_gt_calpha3_frame_positions'] = ret['right_gt_calpha3_frame_positions'] * ret['right_gt_calpha3_frame_position_exists'][...,None]
      ret['left_forth_atom_rel_pos'] = ret['left_gt_calpha3_frame_positions'] * ret['left_gt_calpha3_frame_position_exists'][...,None]

  return ret


def atom37_to_frames(aatype, all_atom_positions, all_atom_mask):
  aatype_in_shape = aatype.shape
  device = aatype.device

  aatype = torch.reshape(aatype, [-1])
  all_atom_positions = torch.reshape(all_atom_positions, [-1, 37, 3])
  all_atom_mask = torch.reshape(all_atom_mask, [-1, 37])

  # shape (N, 8, 3)
  residx_rigidgroup_base_atom37_idx = batched_select(
      torch.tensor(residue_constants.restype_rigidgroup_base_atom37_idx, device=device), aatype)

  base_atom_pos = batched_select(
      all_atom_positions,
      residx_rigidgroup_base_atom37_idx, batch_dims=1)

  gt_frames = r3.rigids_from_3_points(base_atom_pos[:, :, 0, :], base_atom_pos[:, :, 1, :], point_on_xy_plane=base_atom_pos[:, :, 2, :])

  # (N, 8)
  group_exists = batched_select(torch.tensor(residue_constants.restype_rigidgroup_mask, device=device), aatype)

  gt_atoms_exist = batched_select(
      all_atom_mask,
      residx_rigidgroup_base_atom37_idx,
      batch_dims=1)

  gt_exists = torch.logical_and(torch.all(gt_atoms_exist, dim=-1), group_exists)

  residx_rigidgroup_is_ambiguous = batched_select(torch.tensor(residue_constants.restype_rigidgroup_is_ambiguous, device=device), aatype)
  residx_rigidgroup_ambiguity_rot = batched_select(torch.tensor(residue_constants.restype_rigidgroup_rots, device=aatype.device), aatype)
  
  alt_gt_frames = r3.rigids_mul_rots(gt_frames, residx_rigidgroup_ambiguity_rot)

  gt_frames= (torch.reshape(gt_frames[0], aatype_in_shape + (8, 3, 3)),torch.reshape(gt_frames[1], aatype_in_shape + (8, 3)))
  gt_exists = torch.reshape(gt_exists, aatype_in_shape + (8,))
  group_exists = torch.reshape(group_exists, aatype_in_shape + (8,))
  residx_rigidgroup_is_ambiguous = torch.reshape(residx_rigidgroup_is_ambiguous, aatype_in_shape + (8,))
  alt_gt_frames= (torch.reshape(alt_gt_frames[0], aatype_in_shape + (8, 3, 3)), torch.reshape(alt_gt_frames[1], aatype_in_shape + (8,3)))

  return {
      'rigidgroups_gt_frames': gt_frames,
      'rigidgroups_gt_exists': gt_exists, 
      'rigidgroups_group_exists': group_exists,
      'rigidgroups_group_is_ambiguous': residx_rigidgroup_is_ambiguous,
      'rigidgroups_alt_gt_frames': alt_gt_frames
  }

def atom37_to_torsion_angles(aatype, all_atom_pos, all_atom_mask):
    num_batch, num_res = aatype.shape
    device = aatype.device

    prev_all_atom_pos = F.pad(all_atom_pos[:, :-1], [0, 0, 0, 0, 1, 0])
    prev_all_atom_mask = F.pad(all_atom_mask[:, :-1], [0, 0, 1, 0])

    # shape (B, N, atoms=4, xyz=3)
    pre_omega_atom_pos = torch.cat([
        prev_all_atom_pos[:, :, 1:3, :],  # prev CA, C
        all_atom_pos[:, :, 0:2,:]  # this N, CA
        ], dim=-2)
    phi_atom_pos = torch.cat([
        prev_all_atom_pos[:, :, 2:3, :],  # prev C
        all_atom_pos[:, :, 0:3, :]  # this N, CA, C
        ], dim=-2)
    psi_atom_pos = torch.cat([
        all_atom_pos[:, :, 0:3, :],  # this N, CA, C
        all_atom_pos[:, :, 4:5, :]  # this O
        ], dim=-2)

    # Shape [batch, num_res]
    pre_omega_mask = torch.logical_and(
            torch.all(prev_all_atom_mask[:, :, 1:3], dim=-1),  # prev CA, C
            torch.all(all_atom_mask[:, :, 0:2], dim=-1))  # this N, CA
    phi_mask = torch.logical_and(
            prev_all_atom_mask[:, :, 2], # prev C
            torch.all(all_atom_mask[:, :, 0:3], dim=-1))  # this N, CA, C
    psi_mask = torch.logical_and(
        torch.all(all_atom_mask[:, :, 0:3], dim=-1),# this N, CA, C
        all_atom_mask[:, :, 4])  # this O


    # Compute the table of chi angle indices. Shape: [restypes, chis=4, atoms=4].
    # Select atoms to compute chis. Shape: [batch, num_res, chis=4, atoms=4].
    atom_indices = batched_select(torch.tensor(residue_constants.chi_angles_atom_indices, device=device), aatype)
    # Gather atom positions. Shape: [batch, num_res, chis=4, atoms=4, xyz=3].
    chis_atom_pos = batched_select(all_atom_pos, atom_indices, batch_dims=2)

    # Compute the chi angle mask. I.e. which chis angles exist according to the
    # aatype. Shape [batch, num_res, chis=4].
    chis_mask = batched_select(torch.tensor(residue_constants.chi_angles_mask, device=device), aatype)

    # Gather the chi angle atoms mask. Shape: [batch, num_res, chis=4, atoms=4].
    chi_angle_atoms_mask = batched_select(all_atom_mask, atom_indices, batch_dims=2)

    # Check if all 4 chi angle atoms were set. Shape: [batch, num_res, chis=4].
    chi_angle_atoms_mask = torch.all(chi_angle_atoms_mask, dim=-1)
    chis_mask = torch.logical_and(chis_mask, chi_angle_atoms_mask)

    # Shape (B, N, torsions=7, atoms=4, xyz=3)
    torsions_atom_pos = torch.cat([
        pre_omega_atom_pos[:, :, None, :, :],
        phi_atom_pos[:, :, None, :, :],
        psi_atom_pos[:, :, None, :, :],
        chis_atom_pos], dim=2)
    
    # shape (B, N, torsions=7)
    torsion_angles_mask = torch.cat(
        [pre_omega_mask[:, :, None],
         phi_mask[:, :, None],
         psi_mask[:, :, None],
         chis_mask
        ], dim=2)

    # r3.Rigids (B, N, torsions=7)
    torsion_frames = r3.rigids_from_3_points(
            torsions_atom_pos[:, :, :, 1, :],
            torsions_atom_pos[:, :, :, 2, :],
            torsions_atom_pos[:, :, :, 0, :])

    # r3.Vecs (B, N, torsions=7)
    forth_atom_rel_pos = r3.rigids_mul_vecs(
        r3.invert_rigids(torsion_frames),
        torsions_atom_pos[:, :, :, 3, :])

    # np.ndarray (B, N, torsions=7, sincos=2)
    torsion_angles_sin_cos = torch.stack(
        [forth_atom_rel_pos[...,2], forth_atom_rel_pos[...,1]], dim=-1)
    torsion_angles_sin_cos = torsion_angles_sin_cos / torch.sqrt(
        torch.sum(torch.square(torsion_angles_sin_cos), dim=-1, keepdims=True)
        + 1e-8)
    
    chi_is_ambiguous = batched_select(
        torch.tensor(residue_constants.chi_pi_periodic, device=device), aatype)
    mirror_torsion_angles = torch.cat(
        [torch.ones([num_batch, num_res, 3], device=device),
         1.0 - 2.0 * chi_is_ambiguous], dim=-1)
    alt_torsion_angles_sin_cos = (
        torsion_angles_sin_cos * mirror_torsion_angles[:, :, :, None])
    
    return {
        'torsion_angles_sin_cos': torsion_angles_sin_cos,  # (B, N, 7, 2)
        'alt_torsion_angles_sin_cos': alt_torsion_angles_sin_cos,  # (B, N, 7, 2)
        'torsion_angles_mask': torsion_angles_mask  # (B, N, 7)
    }

