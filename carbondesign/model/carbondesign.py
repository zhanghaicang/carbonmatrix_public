import functools
import logging
import random

import torch
from torch import nn

from carbondesign.common import residue_constants
from carbondesign.model.seqformer import EmbeddingAndSeqformer
from carbondesign.model.head import HeaderBuilder

class CarbonDesignIteration(nn.Module):
    def __init__(self, config):
        super().__init__()

        c = config

        self.seqformer = EmbeddingAndSeqformer(c.embeddings_and_seqformer)

        self.heads = HeaderBuilder.build(
                c.heads,
                seq_channel=c.embeddings_and_seqformer.seq_channel,
                pair_channel=c.embeddings_and_seqformer.pair_channel,
                parent=self)

        self.config = config

    def forward(self, batch, compute_loss = False):
        c = self.config

        seq_act, pair_act = self.seqformer(batch)

        representations = {'pair': pair_act, 'seq': seq_act}

        ret = {}

        ret['representations'] = representations

        ret['heads'] = {}

        for name, module, options in self.heads:
            if compute_loss or name == 'seqhead':
                value = module(ret['heads'], representations, batch)
                if value is not None:
                    ret['heads'][name] = value

        return ret

class CarbonDesign(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.impl = CarbonDesignIteration(config)

        self.config = config


    def forward(self, batch, compute_loss=False):
        c = self.config 

        seq = batch['seq']

        batch_size, num_residues, device = *seq.shape[:2], seq.device

        def get_prev(ret):
            new_prev = {
                    #'prev_pos': ret['heads']['folding']['final_atom_positions'].detach(),
                    'prev_seq': ret['representations']['seq'].detach(),
                    'prev_pair': ret['representations']['pair'].detach(),
                    }
            return new_prev
            
        if c.num_recycle > 0:
            emb_config = c.embeddings_and_seqformer
            prev = {
                    #'prev_pos': torch.zeros([batch_size, num_residues, c.num_atom, 3], device=device),
                    'prev_seq': torch.zeros([batch_size, num_residues, emb_config.seq_channel], device=device),
                    'prev_pair': torch.zeros([batch_size, num_residues, num_residues, emb_config.pair_channel], device=device),
            }
            batch.update(prev)
        
        if self.training:
            num_recycle = random.randint(0, c.num_recycle)
        else:
            num_recycle = c.num_recycle
        with torch.no_grad():
            batch.update(is_recycling=True)
            for i in range(num_recycle):
                ret = self.impl(batch, compute_loss=False)
                batch.update(ret)

                prev = get_prev(ret)
                batch.update(prev)

        batch.update(is_recycling=False)
        ret = self.impl(batch, compute_loss=compute_loss)
        
        return ret
