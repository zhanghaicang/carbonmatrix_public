import torch

from esm.pretrained import load_model_and_alphabet_local
from einops import rearrange

from torch.nn import functional as F

from carbondesign.common import residue_constants

import numpy as np

ESM_EMBED_LAYER = 33
ESM_EMBED_DIM = 1280

# adapted from https://github.com/facebookresearch/esm
_extractor_dict = {}

class ESMEmbeddingExtractor:
    def __init__(self, model_path, repr_layer=None):
        self.model, self.alphabet = load_model_and_alphabet_local(model_path)
        self.model.eval()

        if repr_layer is None:
            repr_layer = ESM_EMBED_LAYER

        self.repr_layer = repr_layer

        convert_aatype_table = np.zeros((len(residue_constants.restype_order_with_x),))
        for a, i in residue_constants.restype_order_with_x.items():
            convert_aatype_table[i] = self.alphabet.tok_to_idx[a]
        self.convert_aatype_table = torch.tensor(convert_aatype_table, dtype=torch.float32).reshape((-1,1))
        self.device = None
    
    def __call__(self, label_seqs, device=None):
        return self.extract(label_seqs, device=device)

    def extract(self, label_seqs, device=None):
        if self.device is None and device is not None:
            self.device = device
            self.model.to(device=device)
        
        with torch.no_grad():
            batch_tokens = F.embedding(label_seqs, self.convert_aatype_table.to(device=device)).to(dtype=torch.int64)
            batch_tokens = torch.squeeze(batch_tokens, dim=-1)
            bs, seq_len = batch_tokens.shape[:2]
            batch_tokens = torch.cat([
                torch.full((bs, 1), self.alphabet.cls_idx, dtype=batch_tokens.dtype, device=batch_tokens.device),
                batch_tokens,
                torch.full((bs, 1), self.alphabet.eos_idx, dtype=batch_tokens.dtype, device=batch_tokens.device),
                ], dim=1)
            results = self.model(batch_tokens, repr_layers=[self.repr_layer])
            
            single = results['representations'][self.repr_layer][:,1:-1]

        return single

    @staticmethod
    def get(model_path, repr_layer=None, device=None):
        global _extractor_dict

        if model_path not in _extractor_dict:
            obj = ESMEmbeddingExtractor(model_path, repr_layer=repr_layer)
            if device is not None:
                obj.model.to(device=device)
            _extractor_dict[model_path] = obj
        return _extractor_dict[model_path]
