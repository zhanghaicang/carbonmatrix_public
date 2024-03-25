import logging

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

from carbondesign.model.utils import batched_select
from carbondesign.model import r3
from carbondesign.model import quat_affine
from carbondesign.model.common_modules import(
        Linear,
        LayerNorm)
from carbondesign.model.seqformer import SpatialDepthWiseInception
from carbondesign.model.sidechain import MultiRigidSidechain
from carbondesign.common import residue_constants

logger = logging.getLogger(__name__)

class StructureModule(nn.Module):
    def __init__(self, config, num_in_seq_channel, num_in_pair_channel):
        super().__init__()

        c = config

        self.init_seq_layer_norm = LayerNorm(num_in_seq_channel)
        self.proj_init_seq_act = Linear(num_in_seq_channel, c.num_channel, init='linear')

        self.sidechain_module = MultiRigidSidechain(config, num_in_seq_channel)

        self.config = c

    def forward(self, representations, batch):
        c = self.config

        seq_act = representations['seq']
        b, n, device = *seq_act.shape[:2], seq_act.device

        # (b, 21, n)
        seq = torch.tile(torch.arange(21, device=device), [b * n, 1])
        seq = rearrange(seq, '(b n) c -> (b c) n', b=b)

        seq_act = self.proj_init_seq_act(seq_act)
        seq_act = self.init_seq_layer_norm(seq_act)

        outputs = dict(sidechains=[], traj=[])

        backbone_frame = batch['rigidgroups_gt_frames']
        rotations, translations  = tuple(map(lambda x : x[:, :, 0], backbone_frame))
        
        # (b, n, 3 * 3)
        # (b, n, 3)
        rotations = torch.tile(rotations[:,None], [1, 21, 1, 1, 1])
        translations = torch.tile(translations[:,None], [1, 21, 1, 1])
        rotations = rearrange(rotations, 'b a n t r -> (b a) n t r')
        translations = rearrange(translations, 'b a n r -> (b a) n r')

        sidechains = self.sidechain_module(
                    seq,
                    (rotations, translations),
                    seq_act, compute_atom_pos=True)

        outputs['sidechains'] = sidechains

        outputs['representations'] = {'structure_module': seq_act}

        atom14_positions = outputs['sidechains']['atom_pos']

        residx_atom37_to_atom14 = batched_select(torch.tensor(residue_constants.restype_atom37_to_atom14, device=device), seq)
        atom37_positions = batched_select(atom14_positions, residx_atom37_to_atom14, batch_dims=2)

        outputs['final_atom14_positions'] = rearrange(atom14_positions, '(b a) l n r -> b l a n r', a=21)
        outputs['final_atom_positions'] = atom37_positions 


        return outputs
