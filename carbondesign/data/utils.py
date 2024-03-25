
import logging
import os

import numpy as np

from Bio.PDB.Chain import Chain
from Bio.PDB.Atom import Atom
from Bio.PDB.Residue import Residue
from Bio.PDB.Model import Model as PDBModel
from Bio.PDB.PDBIO import PDBIO

import torch
from torch.nn import functional as F

from carbondesign.common import residue_constants

def pad_for_batch(items, batch_length, dtype):
    batch = []
    if dtype == 'seq':
        for seq in items:
            z = torch.ones(batch_length - seq.shape[0], dtype=seq.dtype) * residue_constants.unk_restype_index
            c = torch.cat((seq, z), dim=0)
            batch.append(c)
    elif dtype == 'msk':
        # Mask sequences (1 if present, 0 if absent) are padded with 0s
        for msk in items:
            z = torch.zeros(batch_length - msk.shape[0], dtype=msk.dtype)
            c = torch.cat((msk, z), dim=0)
            batch.append(c)
    elif dtype == "crd":
        for item in items:
            z = torch.zeros((batch_length - item.shape[0],  item.shape[-2], item.shape[-1]), dtype=item.dtype)
            c = torch.cat((item, z), dim=0)
            batch.append(c)
    elif dtype == "crd_msk":
        for item in items:
            z = torch.zeros((batch_length - item.shape[0],  item.shape[-1]), dtype=item.dtype)
            c = torch.cat((item, z), dim=0)
            batch.append(c)
    elif dtype == "ebd":
        for item in items:
            z = torch.zeros((batch_length - item.shape[0],  item.shape[-1]), dtype=item.dtype,
device = item.device)
            c = torch.cat((item, z), dim=0)
            batch.append(c)
    elif dtype == "pair":
        for item in items:
            c = F.pad(item, (0, 0, 0, batch_length - item.shape[-2], 0, batch_length - item.shape[-2]))
            batch.append(c)
    else:
        raise ValueError('Not implemented yet!')
    batch = torch.stack(batch, dim=0)
    return batch

def weights_from_file(filename):
    if filename:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in filter(lambda x: len(x)>0, map(lambda x: x.strip(), f)):
                items = line.split()
                yield float(items[0])

def embedding_get_labels(name, mat):
    if name == 'token':
        return [residue_constants.restypes_with_x[i if i < len(residue_constants.restypes_with_x) else -1]
                for i in range(mat.shape[0])]
    return None

def pdb_save(step, batch, headers, prefix='/tmp', is_training=False):
    
    for x, pid in enumerate(batch['name']):
        str_seq = batch['str_heavy_seq'][x] + batch['str_light_seq'][x]
        heavy_len = len(batch['str_heavy_seq'][x])
        N = len(str_seq)
        #aatype = batch['seq'][x,...].numpy()
        aatype = np.array([residue_constants.restype_order_with_x.get(aa, residue_constants.unk_restype_index) for aa in str_seq])
        features = dict(aatype=aatype, residue_index=np.arange(N), heavy_len=heavy_len)

        if is_training:
            p = os.path.join(prefix, '{}_{}_{}.pdb'.format(pid, step, x))
        else:
            p = os.path.join(prefix, f'{pid}.pdb')

        with open(p, 'w') as f:
            coords = headers['folding']['final_atom_positions'].detach().cpu()  # (b l c d)
            _, _, num_atoms, _ = coords.shape
            coord_mask = np.asarray([residue_constants.restype_atom14_mask[restype][:num_atoms] for restype in aatype])

            result = dict(structure_module=dict(
                final_atom_mask = coord_mask,
                final_atom_positions = coords[x,:N].numpy()))
            prot = protein.from_prediction(features=features, result=result)
            f.write(protein.to_pdb(prot))

            #logging.debug('step: {}/{} length: {}/{} PDB save: {}'.format(step, x, masked_seq_len, len(str_seq), pid))

            #torsions = headers['folding']['traj'][-1]['torsions_sin_cos'][x,:len(str_seq)]
            #np.savez(os.path.join(prefix, f'{pid}_{step}_{x}.npz'), torsions = torsions.detach().cpu().numpy())

            if 'coord' in batch:
                if is_training:
                    p = os.path.join(prefix, '{}_{}_{}_gt.pdb'.format(pid, step, x))
                else:
                    p = os.path.join(prefix, f'{pid}_gt.pdb')

                with open(p, 'w') as f:
                    coord_mask = batch['coord_mask'].detach().cpu()
                    coords = batch['coord'].detach().cpu()
                    result = dict(structure_module=dict(
                        final_atom_mask = coord_mask[x,...].numpy(),
                        final_atom_positions = coords[x,...].numpy()))
                    prot = protein.from_prediction(features=features, result=result)
                    f.write(protein.to_pdb(prot))
                    #logging.debug('step: {}/{} length: {}/{} PDB save: {} (groundtruth)'.format(step, x, masked_seq_len, len(str_seq), pid))

def make_chain(aa_types, coords, chain_id, coord_mask=None):
    chain = Chain(chain_id)

    num_atoms = 5
    serial_number = 0
    if coord_mask is None:
        coord_mask = np.ones((len(aa_types),num_atoms), dtype=np.bool_)

    def make_residue(i, aatype, coord):
        nonlocal serial_number
        
        resname = residue_constants.restype_1to3.get(aatype, 'UNK')
        residue = Residue(id=(' ', i, ' '), resname=resname, segid='')
        for j, atom_name in enumerate(residue_constants.restype_name_to_atom14_names[resname][:num_atoms]):
            if atom_name == '' or coord_mask[i, j] == False:
                continue
            
            atom = Atom(name=atom_name, 
                    coord=coord[j],
                    bfactor=0, occupancy=1, altloc=' ',
                    fullname=str(f'{atom_name:<4s}'),
                    serial_number=serial_number, element=atom_name[:1])
            residue.add(atom)

            serial_number += 1

        return residue

    for i, (aa, coord) in enumerate(zip(aa_types, coords)):
        chain.add(make_residue(i, aa, coord))

    return chain

def save_ig_pdb(str_heavy_seq, str_light_seq, coord, pdb_path):
    assert len(str_heavy_seq) + len(str_light_seq) == coord.shape[0]

    heavy_chain = make_chain(str_heavy_seq, coord[:len(str_heavy_seq)], 'H')
    light_chain = make_chain(str_light_seq, coord[len(str_heavy_seq):], 'L')

    model = PDBModel(id=0)
    model.add(heavy_chain)
    model.add(light_chain)
    
    pdb = PDBIO()
    pdb.set_structure(model)
    pdb.save(pdb_path)

def save_general_pdb(str_seq, coord, pdb_path, coord_mask=None):
    assert len(str_seq) == coord.shape[0]

    chain = make_chain(str_seq, coord[:len(str_seq)], 'A', coord_mask)

    model = PDBModel(id=0)
    model.add(chain)
    
    pdb = PDBIO()
    pdb.set_structure(model)
    pdb.save(pdb_path)
