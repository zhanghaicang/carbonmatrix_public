import sys
import functools
import logging
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

from carbondesign.common import residue_constants
from carbondesign.model import atom as functional, folding
from carbondesign.model.common_modules import(
        Linear,
        LayerNorm)
from carbondesign.model.utils import squared_difference

logger = logging.getLogger(__name__)

class SeqHead(nn.Module):
    def __init__(self, config, num_in_channel):
        super().__init__()

        c = config
    
        self.norm = LayerNorm(num_in_channel)
        self.proj = Linear(num_in_channel, len(residue_constants.restypes_with_x), init='final')

        self.config = config

    def forward(self, headers, representations, batch):
        x = representations['seq']
        x = self.norm(x)
        logits = self.proj(x)
        
        return dict(logits=logits)

class PairHead(nn.Module):
    def __init__(self, config, num_in_channel):
        super().__init__()

        c = config
        num_token = len(residue_constants.restypes_with_x)
        self.norm = LayerNorm(num_in_channel)
        self.proj = Linear(num_in_channel, num_token * num_token, init='final')

        self.config = config

        self.num_token = num_token

    def forward(self, headers, representations, batch):
        x = representations['pair']
        x = self.norm(x)

        logits = self.proj(x)
        #aatype = torch.argmax(logits, dim=-1)
        logits = rearrange(logits, 'b i j (a c) -> b i j a c', a = self.num_token)
        logits = rearrange(0.5 * (logits + rearrange(logits, 'b i j a c -> b j i c a')), 'b i j a c -> b i j (a c)')

        return dict(logits=logits)

class FoldingHead(nn.Module):
    """Head to predict 3d struct.
    """
    def __init__(self, config, num_in_seq_channel, num_in_pair_channel):
        super().__init__()
        self.struct_module = folding.StructureModule(config, num_in_seq_channel, num_in_pair_channel)

        self.config = config

    def forward(self, headers, representations, batch):
        return self.struct_module(representations, batch)

class HeaderBuilder:
    @staticmethod
    def build(config, seq_channel, pair_channel, parent):
        head_factory = OrderedDict(
                structure_module = functools.partial(FoldingHead, num_in_seq_channel=seq_channel, num_in_pair_channel=pair_channel),
                seqhead = functools.partial(SeqHead, num_in_channel=seq_channel),
                pairhead = functools.partial(PairHead, num_in_channel=pair_channel),
                )
        def gen():
            for head_name, h in head_factory.items():
                if head_name not in config:
                    continue
                head_config = config[head_name]
                
                head = h(config=head_config)

                if isinstance(parent, nn.Module):
                    parent.add_module(head_name, head)
                
                if head_name == 'structure_module':
                    head_name = 'folding'
                
                yield head_name, head, head_config

        return list(gen())
