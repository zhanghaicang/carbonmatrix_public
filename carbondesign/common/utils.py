import numpy as np

from carbondesign.common import residue_constants

def str_seq_to_index(str_seq, mapping=residue_constants.restype_order_with_x, map_unknown_to_x=True):
    seq = []
    for aa in str_seq:
      if aa not in mapping and not map_unknown_to_x:
          raise ValueError(f'Invalid character in the sequence: {aa}')
      seq.append(mapping.get(aa, mapping['X']))

    return np.array(seq)

def index_to_str_seq(index, mapping=residue_constants.restypes_with_x, map_unknown_to_x=True):
    str_seq = [mapping[i] for i in index]
    return ''.join(str_seq)
