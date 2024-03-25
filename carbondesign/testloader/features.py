import torch
from torch.nn import functional as F

import ml_collections
from einops import rearrange

from carbondesign.model.features import register_feature, _feats_fn 
from carbondesign.model import r3 
from carbondesign.common import residue_constants
from carbondesign.model.utils import (
        batched_select, 
        squared_difference)

from carbondesign.testloader import geometry

def virtual_beta_fn(all_atom_positions, all_atom_masks):
    assert all_atom_positions.shape[-2] == len(residue_constants.atom_order)

    N_idx = residue_constants.atom_order['N']
    CA_idx = residue_constants.atom_order['CA']
    C_idx = residue_constants.atom_order['C']

    b = all_atom_positions[..., CA_idx, :] - all_atom_positions[..., N_idx, :]
    c = all_atom_positions[..., C_idx, :] - all_atom_positions[..., CA_idx, :]
    a = torch.cross(b, c, dim=-1)

    pseudo_beta = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + all_atom_positions[..., CA_idx, :]
    

    if all_atom_masks is not None:
        pseudo_beta_mask = torch.all(torch.stack([
            all_atom_masks[...,N_idx],
            all_atom_masks[...,CA_idx],
            all_atom_masks[...,C_idx]], dim=-1), dim=-1)

        return pseudo_beta, pseudo_beta_mask

    return pseudo_beta

@register_feature
def make_atom14_alt_gt_positions(batch, ):
    assert 'atom14_gt_positions' in batch and 'atom14_gt_exists' in batch
    device = batch['seq'].device

    restype_atom_swap_index = batched_select(torch.tensor(residue_constants.restype_ambiguous_atoms_swap_index, device=device), batch['seq'])
    batch['atom14_alt_gt_positions'] = batched_select(batch['atom14_gt_positions'], restype_atom_swap_index, batch_dims=2)
    batch['atom14_alt_gt_exists'] = batched_select(batch['atom14_gt_exists'], restype_atom_swap_index, batch_dims=2)

    return batch

@register_feature
def make_pseudo_beta(batch, ):
    if 'atom37_gt_positions' not in batch:
        batch = make_atom37_positions(batch)
    
    batch['pseudo_beta'], batch['pseudo_beta_mask'] = virtual_beta_fn(batch['atom37_gt_positions'], batch['atom37_gt_exists'])

    return batch

@register_feature
def make_gt_frames(batch, ):
    if 'atom37_gt_positions' not in batch:
        batch = make_atom37_positions(batch)

    batch.update(
            geometry.atom37_to_frames(batch['seq'], batch['atom37_gt_positions'], batch['atom37_gt_exists']))
    
    return batch

@register_feature
def make_calpha3_frames(batch, ):
    calpha_pos = batch['atom37_gt_positions'][:,:,1]
    calpha_mask = batch['atom37_gt_exists'][:,:,1]
    
    batch.update(geometry.calpha3_to_frames(calpha_pos, calpha_mask))

    return batch

@register_feature
def make_torsion_angles(batch, ):
    if 'atom37_gt_positions' not in batch:
        batch = make_atom37_positions(batch)
    
    batch.update(
            geometry.atom37_to_torsion_angles(batch['seq'], batch['atom37_gt_positions'], batch['atom37_gt_exists']))

    return batch

@register_feature
def make_geometry_features(batch, config, ):
    c = ml_collections.ConfigDict(config)
    # residue depth
    ca_center = batch['geo_global'][:,None]
    atom37_positions = batch['atom37_gt_positions']
    pseudo_beta = batch['pseudo_beta']
    ca_pos = atom37_positions[:,:,1]

    device = atom37_positions.device

    # (b, l, 1)
    #rd = torch.sqrt(torch.sum(squared_difference(atom37_positions[:,:,1], ca_center), dim=-1, keepdims=True))
    #ca_center_frame = r3.rigids_from_3_points(ca_center, atom37_positions[:,:,1], pseudo_beta)
    #center_ca_cb_angle = r3.rigids_mul_vecs(r3.invert_rigids(ca_center_frame), pseudo_beta)[:,:,:2]
    
    # determine the orientation of Cb
    # angles of CA_center->CA->Cb
    
    # distance on backbones
    breaks = torch.linspace(c.first_break, c.last_break, steps=c.num_bins-1, device=device)
    sq_breaks = torch.square(breaks)
    
    ca_ca_dist2 = r3.point_square_distance(ca_pos[:,None], ca_pos[:,:,None])
    ca_ca_dis_bin = torch.sum(ca_ca_dist2[...,None] > sq_breaks, axis=-1)
    ca_ca_dis_mask = torch.le(ca_ca_dist2, c.last_break ** 2) 
   
    batch['ca_ca_dis_one_hot'] = F.one_hot(ca_ca_dis_bin, num_classes=c.num_bins).to(dtype=ca_ca_dist2.dtype)
    batch['ca_ca_dis_mask'] = ca_ca_dis_mask

    return batch

@register_feature
def make_proteinmpnn_features(batch, config, ):
    c = ml_collections.ConfigDict(config)

    # residue depth
    pseudo_beta = batch['pseudo_beta']
    bb_positions = torch.cat([batch['atom14_gt_positions'][:,:,:4], batch['pseudo_beta'][:,:,None]], dim=2)
    bb_mask = torch.cat([batch['atom14_gt_exists'][:,:,:4], batch['pseudo_beta_mask'][:,:,None]], dim=2)

    device = bb_positions.device
    
    # distance on backbones
    breaks = torch.linspace(c.first_break, c.last_break, steps=c.num_bins-1, device=device)
    sq_breaks = torch.square(breaks)
   
    # (b, l, n, 3) -> (b, l, l, n, n)
    # (b, l, 1, n, 1), (b, 1, l, 1, n)
    dist2 = r3.point_square_distance(
            rearrange(bb_positions, 'b l n r -> b l () n () r'),
            rearrange(bb_positions, 'b l n r -> b () l () n r'))
    ca_ca_dist2 = dist2[:,:,:,1,1]

    dist2 = rearrange(dist2, 'b i j m n -> b i j (m n) ()')
    
    num_residues = dist2.shape[1]

    batch['pair_mask'] = torch.logical_and(
            torch.logical_and(batch['mask'][:,None], batch['mask'][:,:,None]),
            torch.logical_and(
                torch.le(ca_ca_dist2, c.last_break ** 2),
                torch.logical_not(torch.diag(torch.ones((num_residues,), device=device, dtype=torch.bool)))))
            
    atom_mask = rearrange(
        rearrange(bb_mask, 'b i a -> b i () a ()') * rearrange(bb_mask, 'b j c -> b () j () c'),
        'b i j a c -> b i j (a c)')
    dist_bin = torch.sum(dist2 > sq_breaks, axis=-1) 

    batch['dist_one_hot'] = rearrange(
            F.one_hot(dist_bin, num_classes=c.num_bins).to(dtype=dist2.dtype) * atom_mask[...,None],
            'b i j n d -> b i j (n d)')

    return batch


def make_atom37_positions(batch):
    device = batch['seq'].device
    assert 'atom14_gt_positions' in batch and 'atom14_gt_exists' in batch
    
    batch['atom37_gt_positions'] = batched_select(batch['atom14_gt_positions'], batch['residx_atom37_to_atom14'], batch_dims=2)
    batch['atom37_gt_exists'] = torch.logical_and(
            batched_select(batch['atom14_gt_exists'], batch['residx_atom37_to_atom14'], batch_dims=2),
            batch['atom37_atom_exists'])

    return batch
