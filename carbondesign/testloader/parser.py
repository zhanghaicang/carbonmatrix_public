from os.path import basename, splitext
from collections import OrderedDict
import numpy as np

from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Chain import Chain as PDBChain
from Bio.PDB.Residue import Residue
from Bio.PDB.vectors import Vector as Vector, calc_dihedral

from carbondesign.common import residue_constants

def renum_chain(orig_chain, struc2seq):
    chain = PDBChain(orig_chain.id)
    for residue in orig_chain:
        idx = struc2seq[residue.id]
        new_residue = Residue(id=(' ', idx, ' '), resname=residue.resname, segid=residue.segid)
        for atom in residue:
            atom.detach_parent()
            new_residue.add(atom)
        chain.add(new_residue)
    return chain

def extract_chain_subset(orig_chain, res_ids):
    chain = PDBChain(orig_chain.id)
    for residue in orig_chain:
        if residue.id in res_ids:
            residue.detach_parent()
            chain.add(residue)
    return chain

def check_backbone_coverage(chain):
    cnt = 0
    for r in chain:
        if r.has_id('N') and r.has_id('CA') and r.has_id('C'):
            cnt += 1
    return cnt / len(chain) 

def has_valid_preomega(prev_r, r):
    return not(prev_r is None or r is None)\
            and prev_r.has_id('CA') and prev_r.has_id('C') and r.has_id('N')\
            and r.has_id('CA') and r['N'] - prev_r['C'] < 2.

def has_valid_phi(prev_r, r):
    return not(prev_r is None or r is None)\
            and prev_r.has_id('C') and r.has_id('N') and r.has_id('CA') and r.has_id('C')\
            and r['N'] - prev_r['C'] < 2.

def has_valid_psi(r):
    return r is not None\
            and r.has_id('N') and r.has_id('CA') and r.has_id('C') and r.has_id('O')


def calc_torsion_4_atoms(a1, a2, a3, a4):
    return calc_dihedral(Vector(a1.coord), Vector(a2.coord), Vector(a3.coord), Vector(a4.coord))
    
def get_continuous_ranges(residues):
    """ Returns ranges of residues which are continuously connected (peptide bond length 1.2-1.45 Å) """
    dists = []
    for res_i in range(len(residues) - 1):
        if 'C' not in residues[res_i] or 'N' not in residues[res_i + 1]:
            dists.append(10.0)
        else:
            dists.append(np.linalg.norm(residues[res_i]['C'].get_coord() -
                residues[res_i + 1]['N'].get_coord()))

    ranges = []
    start_i = 0
    for d_i, d in enumerate(dists):
        if d > 1.45 or d < 1.2:
            ranges.append((start_i, d_i + 1))
            start_i = d_i + 1
        if d_i == len(dists) - 1:
            ranges.append((start_i, len(residues)))

    return ranges

def get_struc2seq_map(chain, seq):

    def _resname(residue):
        return residue_constants.restype_3to1.get(
                residue.resname, residue_constants.restypes_with_x[-1])

    struc_seq = ''.join([_resname(r) for r in chain])
   
    residues = [r for r in chain]
    continuous_ranges = get_continuous_ranges(residues)

    struc2seq, seq2struc = OrderedDict(), OrderedDict()
    seq_start, seq_end  = (-1, 0)
    
    for range_idx, (struc_start, struc_end) in enumerate(continuous_ranges):
        seq_start = seq_end + seq[seq_end:].index(struc_seq[struc_start:struc_end])
        seq_end = seq_start + struc_end - struc_start

        for i in range(struc_start, struc_end):
            struc_id, seq_id = residues[i].id, seq_start + i - struc_start
            struc2seq[struc_id] = seq_id
            seq2struc[seq_id] = struc_id

    return struc2seq, seq2struc

def get_struc2seq_ab_map(chain, seq):

    def _resname(residue):
        return residue_constants.restype_3to1.get(
                residue.resname, residue_constants.restypes_with_x[-1])

    struc_seq = ''.join([_resname(r) for r in chain])
   
    residues = [r for r in chain]
    continuous_ranges = get_continuous_ranges(residues)

    struc2seq, seq2struc = OrderedDict(), OrderedDict()
    seq_start, seq_end  = (-1, 0)
    
    for range_idx, (struc_start, struc_end) in enumerate(continuous_ranges):
        if seq[seq_end:].find(struc_seq[struc_start:struc_end]) == -1:
            if struc_end - struc_start <= 2:
                continue
            seq1 = seq[seq_end:]
            seq2 = struc_seq[struc_start:struc_end]
            _, aligned_struc_seq = align_seq(seq1, seq2)
            assert seq2.find(aligned_struc_seq) != -1
            struc_start += seq2.find(aligned_struc_seq)
            struc_end = struc_start + len(aligned_struc_seq)

        seq_start = seq_end + seq[seq_end:].index(struc_seq[struc_start:struc_end])
        seq_end = seq_start + struc_end - struc_start

        for i in range(struc_start, struc_end):
            struc_id, seq_id = residues[i].id, seq_start + i - struc_start
            struc2seq[struc_id] = seq_id
            seq2struc[seq_id] = struc_id

    return struc2seq, seq2struc

def get_pdb_id(pdb_file_path):
    return splitext(basename(pdb_file_path))[0]

def parse_pdb(pdb_file, model=0):
    parser = PDBParser()
    structure = parser.get_structure(get_pdb_id(pdb_file), pdb_file)
    return structure[model]

def make_struc_feature(residues, str_seq, pdb2seq):
    n = len(str_seq)
    assert n > 0
   
    coords = np.zeros((n, 14, 3), dtype=np.float32)
    coord_mask = np.zeros((n, 14), dtype=np.bool)

    # (pre-omega, phi, psi)
    torsions = np.zeros((n, 3), dtype=np.float32)
    torsion_mask = np.zeros((n, 3), dtype=np.bool)

    current_resseq = None
    residue_plddt, residue_n, atom_plddt, atom_n = 0.0, 0, 0.0, 0
    residue_n, atom_n = 0.0, 0.0

    prev_r = None

    for r in residues:
        res_atom14_list = residue_constants.restype_name_to_atom14_names[r.resname]
        if not r.id in pdb2seq:
            continue
        seq_idx = pdb2seq[r.id]

        for atom in r.get_atoms():
            if atom.id not in res_atom14_list:
                continue
            atom14idx = res_atom14_list.index(atom.id)
            coords[seq_idx][atom14idx] = atom.get_coord()
            coord_mask[seq_idx][atom14idx]= True

        if has_valid_preomega(prev_r, r):
            torsions[seq_idx][0] = calc_torsion_4_atoms(prev_r['CA'], prev_r['C'], r['N'], r['CA'])
            torsion_mask[seq_idx][0] = True

        if has_valid_phi(prev_r, r):
            torsions[seq_idx][1] = calc_torsion_4_atoms(prev_r['C'], r['N'], r['CA'], r['C'])
            torsion_mask[seq_idx][1] = True

        if has_valid_psi(r):
            torsions[seq_idx][2] = calc_torsion_4_atoms(r['N'], r['CA'], r['C'], r['O'])
            torsion_mask[seq_idx][2] = True

        prev_r = r

    return dict(coords=coords, coord_mask=coord_mask, torsions=torsions, torsion_mask=torsion_mask)
