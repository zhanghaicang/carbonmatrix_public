import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

#from torch.nn import LayerNorm

from carbondesign.common import residue_constants


class Linear(nn.Module):
    def __init__(self, input_dim, output_dim, init, bias=True):
        super().__init__()

        self.proj = nn.Linear(input_dim, output_dim, bias=bias)

        assert init in ['gate', 'final', 'attn', 'relu', 'linear']

        if init in ['gate', 'final']:
            nn.init.constant_(self.proj.weight, 0.)
        elif init == 'attn':
            # GlorotUniform
            torch.nn.init.xavier_uniform_(self.proj.weight)
        elif init in ['relu', 'linear']:
            # Relu, He
            # linear, Le cun
            distribution_stddev = 0.87962566103423978
            scale = 2. if init == 'relu' else 1.
            stddev = np.sqrt(scale / input_dim) / distribution_stddev
            nn.init.trunc_normal_(self.proj.weight, mean=0., std=stddev)
        else:
            raise NotImplementedError(f'{init} not Implemented')

        if bias:
            if init == 'gate':
                nn.init.constant_(self.proj.bias, 1.)
            else:
                nn.init.constant_(self.proj.bias, 0.)

    def forward(self, x):
        return self.proj(x)

# the operator torch.sigmoid is very unstable in mlu
class Gate(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.proj1 = Linear(input_dim, output_dim, init='gate', bias=True)
        self.proj2 = Linear(input_dim, output_dim, init='gate', bias=True)

    def forward(self, x):
        logits = torch.stack([
            self.proj1(x),
            self.proj2(x)
            ], dim=-1)
        gated = F.softmax(logits, dim=-1)

        return gated[...,0]


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12, affine=True):
        super().__init__()
        self.hidden_size = (hidden_size,) if isinstance(hidden_size, int) else tuple(hidden_size)
        self.eps = eps
        self.affine = bool(affine)
        if self.affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.weight, self.bias = None, None

    def forward(self, x):
        dims = tuple(-(i + 1) for i in range(len(self.hidden_size)))
        means = x.mean(dims, keepdim=True)
        x_zeromean = x - means
        variances = x_zeromean.pow(2).mean(dims, keepdim=True)
        x = x_zeromean / torch.sqrt(variances + self.eps)
        if self.affine:
            x = (self.weight * x) + self.bias
        return x

def apply_dropout(tensor, rate, is_training, broadcast_dim=None):
    if is_training and rate > 0.0:
        shape = list(tensor.shape)
        if broadcast_dim is not None:
            shape[broadcast_dim] = 1
        with torch.no_grad():
            scale = 1. / (1. - rate)
            keep_rate = torch.full(shape, 1. - rate, dtype=tensor.dtype, device=tensor.device)
            keep = torch.bernoulli(keep_rate)
        return scale * keep * tensor
    else:
        return tensor

def pseudo_beta_fn(aatype, all_atom_positions, all_atom_masks):
    """Create pseudo beta features."""

    is_gly = torch.eq(aatype, residue_constants.restype_order['G'])
    ca_idx = residue_constants.atom_order['CA']
    cb_idx = residue_constants.atom_order['CB']

    pseudo_beta = torch.where(
        #torch.tile(is_gly[..., None], [1] * len(is_gly.shape) + [3]),
        is_gly[...,None],
        all_atom_positions[..., ca_idx, :],
        all_atom_positions[..., cb_idx, :])

    if all_atom_masks is not None:
        pseudo_beta_mask = torch.where(
            is_gly,
            all_atom_masks[..., ca_idx].to(dtype=torch.float32),
            all_atom_masks[..., cb_idx].to(dtype=torch.float32))
        return pseudo_beta, pseudo_beta_mask

    return pseudo_beta

def dgram_from_positions(positions, num_bins, min_bin, max_bin):
    breaks = torch.linspace(min_bin, max_bin, steps=num_bins-1, device=positions.device)
    sq_breaks = torch.square(breaks)

    dist2 = torch.sum(
        torch.square(
            rearrange(positions, 'b l c -> b l () c') -
            rearrange(positions, 'b l c -> b () l c')),
        dim=-1,
        keepdims=True)

    true_bins = torch.sum(dist2 > sq_breaks, axis=-1)
    return F.one_hot(true_bins, num_bins)

def mask_mean(x, mask, dim, eps=1e-10):
    assert x.dim() == 4 and mask.dim() == 3 and dim in [1, 2]
    
    return torch.sum(x * mask[:,:,:,None], dim=dim) / (eps + torch.sum(mask, dim=dim)[:,:,None])
