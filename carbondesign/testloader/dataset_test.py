import os
import functools
import logging
import math
import pathlib
import random

import numpy as np
import torch
from torch.nn import functional as F

from carbondesign.common import residue_constants
from carbondesign.common.utils import str_seq_to_index
from carbondesign.model.features import FeatureBuilder
import carbondesign.testloader.features

from carbondesign.data.utils import pad_for_batch

from Bio.PDB.PDBExceptions import PDBConstructionException
from Bio.PDB import PDBIO
from Bio.PDB.Chain import Chain

from carbondesign.testloader.parser import parse_pdb
from carbondesign.common import residue_constants
logger = logging.getLogger(__file__)

class Cluster(object):
    def __init__(self, names):
        self.names = names
        self.idx = 0
        assert len(names) > 0

    def get_next(self):
        item = self.names[self.idx]
        self.idx += 1
        if self.idx == len(self.names):
            self.idx = 0
        return item

    def __expr__(self):
        return self.names[self.idx]

    def __str__(self):
        return self.names[self.idx]

def parse_cluster(file_name, order=None):
    ret = []
    with open(file_name) as f:
        for line in f:
            items = line.strip().split()
            if order == 'reverse':
                items = items[::-1]
            elif order == 'shuffle':
                random.shuffle(items)
            ret.append(Cluster(names=items))
    return ret

class DistributedDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, rank, word_size):
        super().__init__()
        self.dataset = dataset
        self.rank = rank
        self.word_size = word_size

    def __iter__(self):
        for idx, sample in enumerate(self.dataset):
            if idx % self.word_size == self.rank:
                #logger.info(f'rank= {self.rank} idx= {idx} {sample["name"]}')
                yield sample
    
    def collate_fn(self, *args, **kwargs):
        return self.dataset.collate_fn(*args, **kwargs)


class Dataset(torch.utils.data.IterableDataset):

    def __init__(self, data_dir, name_idx, max_seq_len=None, reduce_num=None, is_cluster_idx=False):
        super().__init__()

        self.data_dir = pathlib.Path(data_dir)
        self.name_idx = name_idx
        self.max_seq_len = max_seq_len
        self.reduce_num = reduce_num
        self.is_cluster_idx = is_cluster_idx

        logger.info(f'dataset size= {len(name_idx)} max_seq_len= {max_seq_len} reduce_num= {reduce_num} is_cluster_idx= {is_cluster_idx}')

        self.epoch_count = 0

    def __len__(self):
        return len(self.name_idx)

    def collate_fn(self, batch, feat_builder=None):
        raise NotImplementedError('dataset collate_fn')

def sample_with_struc(struc_mask, str_len, max_seq_len):
    num_struc = torch.sum(struc_mask)
    if num_struc > 0 and num_struc < str_len:
        struc_start, struc_end = 0, str_len
        while struc_start < str_len and struc_mask[struc_start] == False:
            struc_start += 1
        while struc_end > 0 and struc_mask[struc_end - 1] == False:
            struc_end -= 1
        if struc_end - struc_start > max_seq_len:
            start = random.randint(struc_start, struc_end - max_seq_len)
            end = start + max_seq_len
        else:
            extra = max_seq_len - (struc_end - struc_start)
            left_extra = struc_start - extra // 2 - 10
            right_extra = struc_end + extra // 2 + 10
            start = random.randint(left_extra, right_extra)
            end = start + max_seq_len
            if start < 0:
                start = 0
                end = start + max_seq_len
            elif end > str_len:
                end = str_len
                start = end - max_seq_len
    else:
        start = random.randint(0, str_len - max_seq_len)
        end = start + max_seq_len
    return start, end



def make_feature(structure):
    N = len(structure)

    assert N > 0
    coords = np.zeros((N, 14, 3), dtype=np.float32)
    coord_mask = np.zeros((N, 14), dtype=bool)
    i = 0
    while True:
        i += 1
        residue_id = (' ', i, ' ')
        if residue_id in structure:
            start_num = i
            break
    
    # for i in range(10000):
    #     j = 10000 - i
    #     residue_id = (' ', j, ' ')
    #     if residue_id in structure:
    #         end_num = j
    #         break
    #N = end_num - start_num + 1
    coords = np.zeros((N, 14, 3), dtype=np.float32)
    coord_mask = np.zeros((N, 14), dtype=bool)
    for seq_idx in range(N):
        residue_id = (' ', start_num + seq_idx , ' ')
        if residue_id in structure:
            residue = structure[residue_id]
            if residue.resname not in residue_constants.restype_name_to_atom14_names:
                continue
            res_atom14_list = residue_constants.restype_name_to_atom14_names[residue.resname]

            for atom in residue.get_atoms():
                if atom.id not in ['CA', 'C', 'N', 'O']:
                    continue
                atom14idx = res_atom14_list.index(atom.id)
                coords[seq_idx, atom14idx] = atom.get_coord()
                coord_mask[seq_idx, atom14idx]= True

    feature = dict(
            str_seq='G' * N,
            coords=coords,
            coord_mask=coord_mask)
    return feature


def process(args):
    logging.info(f'processing {args}')
    try:
        struc = parse_pdb(os.path.join(args))
    except PDBConstructionException as e:
        logging.warning('pdb_parse: %s {%s}', name, str(e))
    except Exception as e:
        logging.warning('pdb_parse: %s {%s}', name, str(e))
        raise Exception('...') from e

    assert len(list(struc.get_chains())) == 1

    struc = list(struc.get_chains())[0]
    assert all(map((lambda x : x.get_id()[0] != ' ' or x.get_id()[2] == ' '), struc))

    feature = make_feature(struc)
    return feature

class MonomerDataset(Dataset):
    def __init__(self, data_dir, name_idx, max_seq_len=None, reduce_num=None, is_cluster_idx=False, noise_level=0., noise_sample_ratio=0.):
        super().__init__(data_dir, name_idx, max_seq_len=max_seq_len, reduce_num=reduce_num, is_cluster_idx=is_cluster_idx)

        self.noise_level = noise_level
        self.noise_sample_ratio = noise_sample_ratio
        
    def _get_name_idx(self):
        if self.reduce_num is None:
            return self.name_idx
        
        if self.epoch_count == 0:
            random.seed(2022 + self.epoch_count)
            random.shuffle(self.name_idx)

        start = self.reduce_num * self.epoch_count
        end = start + self.reduce_num
        
        if end > len(self.name_idx):
            start = 0
            end = self.reduce_num
            random.seed(2022 + self.epoch_count)
            random.shuffle(self.name_idx)
            
        logging.info(f'general data: epoch_count={self.epoch_count} reduce_num={self.reduce_num} all={len(self.name_idx)} start={start} end={end} ex={",".join([str(x) for x in self.name_idx[:4]])}')
        
        self.epoch_count += 1
        
        return self.name_idx[start:end]

    def __iter__(self):
        name_idx = self._get_name_idx()
        
        max_seq_len = self.max_seq_len

        for item in name_idx:
            if self.is_cluster_idx:
                name = item.get_next()
            else:
                name = item
            ret = self.get_structure_label_npz(name)
            if max_seq_len is not None:
                str_len  = len(ret['str_seq'])
                if str_len > max_seq_len:
                    start, end = sample_with_struc(ret['atom14_gt_exists'][:,1], str_len, max_seq_len)
                    for k, v in ret.items():
                        if k in ['name', 'geo_global']:
                            continue
                        ret[k] = v[start:end]
                    logger.warn(f'{name} with len= {str_len} to be sliced at postion= {start}')
            yield ret

    def get_structure_label_npz(self, name):
#        struc = np.load(os.path.join(self.data_dir, name + '.npz'))
        
#        coords = torch.from_numpy(struc['coords'])
#        coord_mask = torch.from_numpy(struc['coord_mask'])
        print(self.data_dir)
        print(name)
        
        PDB_feature = process(f"{self.data_dir}/{name}.pdb")
        coords = torch.tensor(PDB_feature['coords'])
        coord_mask = torch.tensor(PDB_feature['coord_mask'])
        if self.noise_level > 0. and self.noise_sample_ratio > 0.:
            if random.random() < self.noise_sample_ratio:
                coords = coords + self.noise_level * torch.randn_like(coords)

        str_seq = str(PDB_feature['seq']) if 'seq' in PDB_feature else str(PDB_feature['str_seq'])

        assert len(str_seq) == coords.shape[0] and len(str_seq) == coord_mask.shape[0] and len(str_seq) > 0

        seq = torch.tensor(str_seq_to_index(str_seq), dtype=torch.int64)

        mask = coord_mask[:, 1]
        aatype_unk_mask = torch.not_equal(seq, residue_constants.unk_restype_index)

        chain_id = torch.zeros((len(str_seq),), dtype=torch.int32)
        
        geo_global = torch.mean(coords[:,1], dim=0) # center of CA atoms

        ret = dict(name=name,
                str_seq = str_seq,
                seq = seq,
                mask = mask,
                aatype_unk_mask=aatype_unk_mask,
                atom14_gt_positions=coords, atom14_gt_exists=coord_mask,
                chain_id = chain_id,
                geo_global = geo_global)

        return ret

    def collate_fn(self, batch, feat_builder=None):
        fields = ('name', 'str_seq', 'seq', 'mask', 'aatype_unk_mask',
                'atom14_gt_positions', 'atom14_gt_exists',
                'chain_id', 'geo_global')
        name, str_seq, seq, mask, aatype_unk_mask, atom14_gt_positions, atom14_gt_exists, chain_id , geo_global =\
                list(zip(*[[b[k] for k in fields] for b in batch]))

        max_len = max(tuple(len(s) for s in str_seq))
        padded_seqs = pad_for_batch(seq, max_len, 'seq')
        padded_masks = pad_for_batch(mask, max_len, 'msk')
        padded_aatype_unk_masks = pad_for_batch(aatype_unk_mask, max_len, 'msk')

        padded_seqs = pad_for_batch(seq, max_len, 'seq')

        padded_atom14_gt_positions = pad_for_batch(atom14_gt_positions, max_len, 'crd')
        padded_atom14_gt_existss = pad_for_batch(atom14_gt_exists, max_len, 'crd_msk')

        padded_chain_id = pad_for_batch(chain_id, max_len, 'msk')

        padded_geo_global = torch.stack(geo_global, dim=0) 

        ret = dict(
		name=name,
                str_seq=str_seq,
                seq=padded_seqs,
                mask=padded_masks,
                aatype_unk_mask=padded_aatype_unk_masks,
                atom14_gt_positions=padded_atom14_gt_positions,
                atom14_gt_exists=padded_atom14_gt_existss,
                chain_id=padded_chain_id,
                geo_global=padded_geo_global,
                data_type = 'monomer',
                )

        if feat_builder:
            #try:
            ret = feat_builder.build(ret)
            #except:
            #    ret = None
            #assert (not is_training or ret is not None)
        return ret

def load(data_dir, name_idx,
        feats=None, data_type='monomer', 
        is_training=False,
        max_seq_len=None, reduce_num=None,
        rank=None, world_size=1,
        is_cluster_idx=False,
        noise_level=0.,
        noise_sample_ratio=0.,
        **kwargs):

    if data_type == 'monomer':
        dataset = MonomerDataset(data_dir, name_idx, max_seq_len=max_seq_len, reduce_num=reduce_num, is_cluster_idx=is_cluster_idx,
                noise_level=noise_level, noise_sample_ratio=noise_sample_ratio)
    else:
        raise NotImplementedError('data type {data_type} not implemented.')

    if rank is not None:
        dataset = DistributedDataset(dataset, rank, world_size)

    kwargs['collate_fn'] =functools.partial(dataset.collate_fn,
            feat_builder=FeatureBuilder(feats, ))

    return torch.utils.data.DataLoader(dataset, **kwargs)
