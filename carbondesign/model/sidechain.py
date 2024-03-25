import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

from carbondesign.model import atom
from carbondesign.model.utils import l2_normalize
from carbondesign.model.common_modules import Linear

class ResNetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.net = nn.Sequential(
                nn.ReLU(),
                Linear(dim, dim, init='relu'),
                nn.ReLU(),
                Linear(dim, dim, init='final'))

    def forward(self, act):
        return act + self.net(act)

class TorsionModule(nn.Module):
    def __init__(self, config, num_in_channel):
        super().__init__()
        c = config

        self.proj_act = nn.Sequential(
                nn.ReLU(),
                Linear(num_in_channel, c.num_channel, init='linear'),
                )
        self.blocks = nn.Sequential(
                *[ResNetBlock(c.num_channel) for _ in range(c.num_residual_block)])

        # (preomega, phi, psi)
        self.projection = Linear(c.num_channel, 21 * 7 * 2, init='linear')

    def forward(self, act):

        act = self.proj_act(act)
        act = self.blocks(act)

        # angles = rearrange(self.projection(F.relu(act)), '... (n d)->... n d', d=2)
        angles = rearrange(self.projection(F.relu(act)), 'b n (a c d) -> (b a) n c d', a=21, d=2)

        return angles

class MultiRigidSidechain(nn.Module):
    def __init__(self, config, num_in_seq_channel):
        super().__init__()
        c = config
        
        self.torsion_module = TorsionModule(c.torsion, c.num_channel)
        
        self.config = config
        
    def forward(self, seq, backb_to_global, seq_act, compute_atom_pos=True):
        
        # Shape: (num_res, 14).
        unnormalized_angles = self.torsion_module(seq_act)
        angles = l2_normalize(unnormalized_angles, dim=-1)

        outputs = {
                'angles_sin_cos': angles, # (N, 7, 2)
                'unnormalized_angles_sin_cos':  unnormalized_angles,  # (N, 7, 2)
                }

        if not compute_atom_pos:
            return outputs
        # (N, 8)
        all_frames_to_global = atom.torsion_angles_to_frames(seq, backb_to_global, angles)

        # (N, 14)
        pred_positions = atom.frames_and_literature_positions_to_atom14_pos(seq, all_frames_to_global)

        outputs.update({
            'atom_pos': pred_positions,  # r3.Vecs (N, 14)
            })
    
        return outputs
