import collections
import functools
import os
from typing import List, Mapping, Tuple

import numpy as np
import tree
import json

restypes = [
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P',
    'S', 'T', 'W', 'Y', 'V'
]
restype_order = {restype: i for i, restype in enumerate(restypes)}
restype_num = len(restypes)
unk_restype_index = restype_num

restypes_with_x = restypes + ['X']
restype_order_with_x = {restype: i for i, restype in enumerate(restypes_with_x)}

restype_1to3 = {
    'A': 'ALA',
    'R': 'ARG',
    'N': 'ASN',
    'D': 'ASP',
    'C': 'CYS',
    'Q': 'GLN',
    'E': 'GLU',
    'G': 'GLY',
    'H': 'HIS',
    'I': 'ILE',
    'L': 'LEU',
    'K': 'LYS',
    'M': 'MET',
    'F': 'PHE',
    'P': 'PRO',
    'S': 'SER',
    'T': 'THR',
    'W': 'TRP',
    'Y': 'TYR',
    'V': 'VAL',
}

restype_3to1 = {v: k for k, v in restype_1to3.items()}

unk_restype = 'UNK'

resnames = [restype_1to3[r] for r in restypes] + [unk_restype]
resname_to_idx = {resname: i for i, resname in enumerate(resnames)}

atom_types = [
    'N', 'CA', 'C', 'CB', 'O', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD',
    'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3',
    'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2',
    'CZ3', 'NZ', 'OXT'
]

atom_order = {atom_type: i for i, atom_type in enumerate(atom_types)}
atom_type_num = len(atom_types)

residue_atoms = {
    'ALA': ['C', 'CA', 'CB', 'N', 'O'],
    'ARG': ['C', 'CA', 'CB', 'CG', 'CD', 'CZ', 'N', 'NE', 'O', 'NH1', 'NH2'],
    'ASP': ['C', 'CA', 'CB', 'CG', 'N', 'O', 'OD1', 'OD2'],
    'ASN': ['C', 'CA', 'CB', 'CG', 'N', 'ND2', 'O', 'OD1'],
    'CYS': ['C', 'CA', 'CB', 'N', 'O', 'SG'],
    'GLU': ['C', 'CA', 'CB', 'CG', 'CD', 'N', 'O', 'OE1', 'OE2'],
    'GLN': ['C', 'CA', 'CB', 'CG', 'CD', 'N', 'NE2', 'O', 'OE1'],
    'GLY': ['C', 'CA', 'N', 'O'],
    'HIS': ['C', 'CA', 'CB', 'CG', 'CD2', 'CE1', 'N', 'ND1', 'NE2', 'O'],
    'ILE': ['C', 'CA', 'CB', 'CG1', 'CG2', 'CD1', 'N', 'O'],
    'LEU': ['C', 'CA', 'CB', 'CG', 'CD1', 'CD2', 'N', 'O'],
    'LYS': ['C', 'CA', 'CB', 'CG', 'CD', 'CE', 'N', 'NZ', 'O'],
    'MET': ['C', 'CA', 'CB', 'CG', 'CE', 'N', 'O', 'SD'],
    'PHE': ['C', 'CA', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'N', 'O'],
    'PRO': ['C', 'CA', 'CB', 'CG', 'CD', 'N', 'O'],
    'SER': ['C', 'CA', 'CB', 'N', 'O', 'OG'],
    'THR': ['C', 'CA', 'CB', 'CG2', 'N', 'O', 'OG1'],
    'TRP': ['C', 'CA', 'CB', 'CG', 'CD1', 'CD2', 'CE2', 'CE3', 'CZ2', 'CZ3',
            'CH2', 'N', 'NE1', 'O'],
    'TYR': ['C', 'CA', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'N', 'O',
            'OH'],
    'VAL': ['C', 'CA', 'CB', 'CG1', 'CG2', 'N', 'O']
}

restype_name_to_atom14_names = {
    'ALA': ['N', 'CA', 'C', 'O', 'CB', '',    '',    '',    '',    '',    '',    '',    '',    ''],
    'ARG': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'NE',  'CZ',  'NH1', 'NH2', '',    '',    ''],
    'ASN': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'OD1', 'ND2', '',    '',    '',    '',    '',    ''],
    'ASP': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'OD1', 'OD2', '',    '',    '',    '',    '',    ''],
    'CYS': ['N', 'CA', 'C', 'O', 'CB', 'SG',  '',    '',    '',    '',    '',    '',    '',    ''],
    'GLN': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'OE1', 'NE2', '',    '',    '',    '',    ''],
    'GLU': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'OE1', 'OE2', '',    '',    '',    '',    ''],
    'GLY': ['N', 'CA', 'C', 'O', '',   '',    '',    '',    '',    '',    '',    '',    '',    ''],
    'HIS': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'ND1', 'CD2', 'CE1', 'NE2', '',    '',    '',    ''],
    'ILE': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1', '',    '',    '',    '',    '',    ''],
    'LEU': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', '',    '',    '',    '',    '',    ''],
    'LYS': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'CE',  'NZ',  '',    '',    '',    '',    ''],
    'MET': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'SD',  'CE',  '',    '',    '',    '',    '',    ''],
    'PHE': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'CE1', 'CE2', 'CZ',  '',    '',    ''],
    'PRO': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  '',    '',    '',    '',    '',    '',    ''],
    'SER': ['N', 'CA', 'C', 'O', 'CB', 'OG',  '',    '',    '',    '',    '',    '',    '',    ''],
    'THR': ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2', '',    '',    '',    '',    '',    '',    ''],
    'TRP': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
    'TYR': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'CE1', 'CE2', 'CZ',  'OH',  '',    ''],
    'VAL': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', '',    '',    '',    '',    '',    '',    ''],
    'UNK': ['',  '',   '',  '',  '',   '',    '',    '',    '',    '',    '',    '',    '',    ''],

}

chi_angles_atoms = {
    'ALA': [],
    # Chi5 in arginine is always 0 +- 5 degrees, so ignore it.
    'ARG': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
            ['CB', 'CG', 'CD', 'NE'], ['CG', 'CD', 'NE', 'CZ']],
    'ASN': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'OD1']],
    'ASP': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'OD1']],
    'CYS': [['N', 'CA', 'CB', 'SG']],
    'GLN': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
            ['CB', 'CG', 'CD', 'OE1']],
    'GLU': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
            ['CB', 'CG', 'CD', 'OE1']],
    'GLY': [],
    'HIS': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'ND1']],
    'ILE': [['N', 'CA', 'CB', 'CG1'], ['CA', 'CB', 'CG1', 'CD1']],
    'LEU': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    'LYS': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
            ['CB', 'CG', 'CD', 'CE'], ['CG', 'CD', 'CE', 'NZ']],
    'MET': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'SD'],
            ['CB', 'CG', 'SD', 'CE']],
    'PHE': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    'PRO': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD']],
    'SER': [['N', 'CA', 'CB', 'OG']],
    'THR': [['N', 'CA', 'CB', 'OG1']],
    'TRP': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    'TYR': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    'VAL': [['N', 'CA', 'CB', 'CG1']],
}

chi_angles_mask = [
    [0.0, 0.0, 0.0, 0.0],  # ALA
    [1.0, 1.0, 1.0, 1.0],  # ARG
    [1.0, 1.0, 0.0, 0.0],  # ASN
    [1.0, 1.0, 0.0, 0.0],  # ASP
    [1.0, 0.0, 0.0, 0.0],  # CYS
    [1.0, 1.0, 1.0, 0.0],  # GLN
    [1.0, 1.0, 1.0, 0.0],  # GLU
    [0.0, 0.0, 0.0, 0.0],  # GLY
    [1.0, 1.0, 0.0, 0.0],  # HIS
    [1.0, 1.0, 0.0, 0.0],  # ILE
    [1.0, 1.0, 0.0, 0.0],  # LEU
    [1.0, 1.0, 1.0, 1.0],  # LYS
    [1.0, 1.0, 1.0, 0.0],  # MET
    [1.0, 1.0, 0.0, 0.0],  # PHE
    [1.0, 1.0, 0.0, 0.0],  # PRO
    [1.0, 0.0, 0.0, 0.0],  # SER
    [1.0, 0.0, 0.0, 0.0],  # THR
    [1.0, 1.0, 0.0, 0.0],  # TRP
    [1.0, 1.0, 0.0, 0.0],  # TYR
    [1.0, 0.0, 0.0, 0.0],  # VAL
    [0.0, 0.0, 0.0, 0.0],  # UNK
]

chi_pi_periodic = [
    [0.0, 0.0, 0.0, 0.0],  # ALA
    [0.0, 0.0, 0.0, 0.0],  # ARG
    [0.0, 0.0, 0.0, 0.0],  # ASN
    [0.0, 1.0, 0.0, 0.0],  # ASP
    [0.0, 0.0, 0.0, 0.0],  # CYS
    [0.0, 0.0, 0.0, 0.0],  # GLN
    [0.0, 0.0, 1.0, 0.0],  # GLU
    [0.0, 0.0, 0.0, 0.0],  # GLY
    [0.0, 0.0, 0.0, 0.0],  # HIS
    [0.0, 0.0, 0.0, 0.0],  # ILE
    [0.0, 0.0, 0.0, 0.0],  # LEU
    [0.0, 0.0, 0.0, 0.0],  # LYS
    [0.0, 0.0, 0.0, 0.0],  # MET
    [0.0, 1.0, 0.0, 0.0],  # PHE
    [0.0, 0.0, 0.0, 0.0],  # PRO
    [0.0, 0.0, 0.0, 0.0],  # SER
    [0.0, 0.0, 0.0, 0.0],  # THR
    [0.0, 0.0, 0.0, 0.0],  # TRP
    [0.0, 1.0, 0.0, 0.0],  # TYR
    [0.0, 0.0, 0.0, 0.0],  # VAL
    [0.0, 0.0, 0.0, 0.0],  # UNK
]

residue_atom_renaming_swaps = {
    'ASP': {'OD1': 'OD2'},
    'GLU': {'OE1': 'OE2'},
    'PHE': {'CD1': 'CD2', 'CE1': 'CE2'},
    'TYR': {'CD1': 'CD2', 'CE1': 'CE2'},
}

def load_rigid_schema():
    new_schema_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
            'new_rigid_schema.json')
    with open(new_schema_path) as f:
        return json.load(f)

rigid_group_atom_positions = load_rigid_schema()

restype_atom14_to_atom37 = np.zeros([21, 14], dtype=np.int32)  # mapping (restype, atom14) --> atom37
restype_atom37_to_atom14 = np.zeros([21, 37], dtype=np.int32)  # mapping (restype, atom37) --> atom14
restype_atom14_mask = np.zeros([21, 14], dtype=np.bool_)
restype_atom37_mask = np.zeros([21, 37], dtype=np.bool_)

def _make_atom14_atom37_map():
    for rt, rt_idx in restype_order.items():
        atom_names = restype_name_to_atom14_names[restype_1to3[rt]]
        for i, name in enumerate(atom_names):
            restype_atom14_to_atom37[rt_idx, i] = atom_order[name] if name else 0
            restype_atom14_mask[rt_idx, i] = 1 if name else 0
        
        atom_name_to_idx14 = {name: i for i, name in enumerate(atom_names)}
        for i, name in enumerate(atom_types):
            restype_atom37_to_atom14[rt_idx, i] = atom_name_to_idx14.get(name, 0)
            restype_atom37_mask[rt_idx, i] = (name in atom_name_to_idx14) 

_make_atom14_atom37_map()

chi_angles_atom_indices = [chi_angles_atoms[restype_1to3[r]] for r in restypes] + [[]]
chi_angles_atom_indices = tree.map_structure(
    lambda atom_name: atom_order[atom_name], chi_angles_atom_indices)
chi_angles_atom_indices = np.array([
    chi_atoms + ([[0, 0, 0, 0]] * (4 - len(chi_atoms)))
    for chi_atoms in chi_angles_atom_indices])

def _make_rigid_transformation_4x4(ex, ey, translation):
    ex_normalized = ex / np.linalg.norm(ex)
    
    ey_normalized = ey - np.dot(ey, ex_normalized) * ex_normalized
    ey_normalized /= np.linalg.norm(ey_normalized)

    eznorm = np.cross(ex_normalized, ey_normalized)
    m = np.stack([ex_normalized, ey_normalized, eznorm, translation]).transpose()
    m = np.concatenate([m, [[0., 0., 0., 1.]]], axis=0)
    
    return m

# rigid groups
restype_atom37_to_rigid_group = np.zeros([21, 37], dtype=np.int32)
restype_atom37_rigid_group_positions = np.zeros([21, 37, 3], dtype=np.float32)
restype_atom14_to_rigid_group = np.zeros([21, 14], dtype=np.int32)
restype_atom14_rigid_group_positions = np.zeros([21, 14, 3], dtype=np.float32)
restype_rigid_group_default_frame = np.zeros([21, 8, 4, 4], dtype=np.float32)

restype_rigidgroup_mask = np.zeros([21, 8], dtype=np.bool_)
restype_rigidgroup_base_atom37_idx = np.zeros([21, 8, 3], np.int32)
restype_rigidgroup_base_atom14_idx = np.zeros([21, 8, 3], np.int32)

def _make_rigid_group_constants():
    for restype, restype_letter in enumerate(restypes):
        resname = restype_1to3[restype_letter]
        for atomname, group_idx, atom_position in rigid_group_atom_positions[resname]:
            atomtype = atom_order[atomname]
            restype_atom37_to_rigid_group[restype, atomtype] = group_idx
            restype_atom37_mask[restype, atomtype] = 1
            restype_atom37_rigid_group_positions[restype, atomtype, :] = atom_position

            atom14idx = restype_name_to_atom14_names[resname].index(atomname)
            restype_atom14_to_rigid_group[restype, atom14idx] = group_idx
            restype_atom14_mask[restype, atom14idx] = 1
            restype_atom14_rigid_group_positions[restype, atom14idx, :] = atom_position

    for restype, restype_letter in enumerate(restypes):
        resname = restype_1to3[restype_letter]
        atom_positions = {name: np.array(pos) for name, _, pos in rigid_group_atom_positions[resname]}

        # backbone to backbone is the identity transform
        restype_rigid_group_default_frame[restype, 0, :, :] = np.eye(4)

        # pre-omega-frame to backbone (currently dummy identity matrix)
        restype_rigid_group_default_frame[restype, 1, :, :] = np.eye(4)

        # phi-frame to backbone
        mat = _make_rigid_transformation_4x4(
                ex=atom_positions['N'] - atom_positions['CA'],
                ey=np.array([1., 0., 0.]),
                translation=atom_positions['N'])
        restype_rigid_group_default_frame[restype, 2, :, :] = mat

        # psi-frame to backbone
        mat = _make_rigid_transformation_4x4(
                ex=atom_positions['C'] - atom_positions['CA'],
                ey=atom_positions['N'] - atom_positions['CA'],
                translation=atom_positions['C'])
        restype_rigid_group_default_frame[restype, 3, :, :] = mat

        # chi1-frame to backbone
        if chi_angles_mask[restype][0]:
            base_atom_names = chi_angles_atoms[resname][0]
            base_atom_positions = [atom_positions[name] for name in base_atom_names]
            mat = _make_rigid_transformation_4x4(
                    ex=base_atom_positions[2] - base_atom_positions[1],
                    ey=base_atom_positions[0] - base_atom_positions[1],
                    translation=base_atom_positions[2])
            restype_rigid_group_default_frame[restype, 4, :, :] = mat

        # chi2-frame to chi1-frame
        # chi3-frame to chi2-frame
        # chi4-frame to chi3-frame
        # luckily all rotation axes for the next frame start at (0,0,0) of the
        # previous frame
        for chi_idx in range(1, 4):
            if chi_angles_mask[restype][chi_idx]:
                axis_end_atom_name = chi_angles_atoms[resname][chi_idx][2]
                axis_end_atom_position = atom_positions[axis_end_atom_name]
                mat = _make_rigid_transformation_4x4(
                        ex=axis_end_atom_position,
                        ey=np.array([-1., 0., 0.]),
                        translation=axis_end_atom_position)
                restype_rigid_group_default_frame[restype, 4 + chi_idx, :, :] = mat
    
    # rigidgroup idx
    def _atom_idx(x):
        return np.array([atom_order.get(i, 0) for i in x])

    # 0: backbone frame
    restype_rigidgroup_base_atom37_idx[:, 0, :] =  _atom_idx(['C', 'CA', 'N']) 
    
    # 3: 'psi-group'
    restype_rigidgroup_base_atom37_idx[:, 3, :] =  _atom_idx(['CA', 'C', 'O']) 

    # 4,5,6,7: 'chi1,2,3,4-group'
    for restype, restype_letter in enumerate(restypes):
        resname = restype_1to3[restype_letter]
        for chi_idx in range(4):
            if chi_angles_mask[restype][chi_idx]:
                atom_names = chi_angles_atoms[resname][chi_idx]
                restype_rigidgroup_base_atom37_idx[restype, chi_idx + 4, :] = _atom_idx(atom_names[1:])

    # Create mask for existing rigid groups.
    restype_rigidgroup_mask[:, 0] = 1
    restype_rigidgroup_mask[:, 3] = 1
    restype_rigidgroup_mask[:20, 4:] = chi_angles_mask[:20]

    return

_make_rigid_group_constants()
  
restype_rigidgroup_is_ambiguous = np.zeros([21, 8], dtype=np.bool_)
restype_rigidgroup_rots = np.tile(np.eye(3, dtype=np.float32), [21, 8, 1, 1])
restype_ambiguous_atoms_swap_index = np.tile(np.arange(14), (21, 1))
restype_atom14_is_ambiguous = np.zeros((21, 14), dtype=np.bool_)

def _make_ambiguous():
    for resname, swap in residue_atom_renaming_swaps.items():
        restype_idx = restype_order_with_x[restype_3to1[resname]] 
        
        chi_idx = int(sum(chi_angles_mask[restype_idx]) - 1)
        restype_rigidgroup_is_ambiguous[restype_idx, chi_idx + 4] = 1
        restype_rigidgroup_rots[restype_idx, chi_idx + 4, 1, 1] = -1
        restype_rigidgroup_rots[restype_idx, chi_idx + 4, 2, 2] = -1
        
        for source_atom_swap, target_atom_swap in swap.items():
            source_index = restype_name_to_atom14_names[resname].index(source_atom_swap)
            target_index = restype_name_to_atom14_names[resname].index(target_atom_swap)
            restype_ambiguous_atoms_swap_index[restype_idx, source_index] = target_index
            restype_ambiguous_atoms_swap_index[restype_idx, target_index] = source_index
            restype_atom14_is_ambiguous[restype_idx, source_index] = 1
            restype_atom14_is_ambiguous[restype_idx, target_index] = 1

_make_ambiguous()


# structure constants
ca_ca = 3.80209737096

van_der_waals_radius = {
    'C': 1.7,
    'N': 1.55,
    'O': 1.52,
    'S': 1.8,
}

Bond = collections.namedtuple(
    'Bond', ['atom1_name', 'atom2_name', 'length', 'stddev'])
BondAngle = collections.namedtuple(
    'BondAngle',
    ['atom1_name', 'atom2_name', 'atom3name', 'angle_rad', 'stddev'])

@functools.lru_cache(maxsize=None)
def load_stereo_chemical_props() -> Tuple[Mapping[str, List[Bond]],
                                          Mapping[str, List[Bond]],
                                          Mapping[str, List[BondAngle]]]:
  stereo_chemical_props_path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)), 'stereo_chemical_props.txt'
  )
  with open(stereo_chemical_props_path, 'rt') as f:
    stereo_chemical_props = f.read()
  lines_iter = iter(stereo_chemical_props.splitlines())
  # Load bond lengths.
  residue_bonds = {}
  next(lines_iter)  # Skip header line.
  for line in lines_iter:
    if line.strip() == '-':
      break
    bond, resname, length, stddev = line.split()
    atom1, atom2 = bond.split('-')
    if resname not in residue_bonds:
      residue_bonds[resname] = []
    residue_bonds[resname].append(
        Bond(atom1, atom2, float(length), float(stddev)))
  residue_bonds['UNK'] = []

  # Load bond angles.
  residue_bond_angles = {}
  next(lines_iter)  # Skip empty line.
  next(lines_iter)  # Skip header line.
  for line in lines_iter:
    if line.strip() == '-':
      break
    bond, resname, angle_degree, stddev_degree = line.split()
    atom1, atom2, atom3 = bond.split('-')
    if resname not in residue_bond_angles:
      residue_bond_angles[resname] = []
    residue_bond_angles[resname].append(
        BondAngle(atom1, atom2, atom3,
                  float(angle_degree) / 180. * np.pi,
                  float(stddev_degree) / 180. * np.pi))
  residue_bond_angles['UNK'] = []

  def make_bond_key(atom1_name, atom2_name):
    """Unique key to lookup bonds."""
    return '-'.join(sorted([atom1_name, atom2_name]))

  # Translate bond angles into distances ("virtual bonds").
  residue_virtual_bonds = {}
  for resname, bond_angles in residue_bond_angles.items():
    # Create a fast lookup dict for bond lengths.
    bond_cache = {}
    for b in residue_bonds[resname]:
      bond_cache[make_bond_key(b.atom1_name, b.atom2_name)] = b
    residue_virtual_bonds[resname] = []
    for ba in bond_angles:
      bond1 = bond_cache[make_bond_key(ba.atom1_name, ba.atom2_name)]
      bond2 = bond_cache[make_bond_key(ba.atom2_name, ba.atom3name)]

      # Compute distance between atom1 and atom3 using the law of cosines
      # c^2 = a^2 + b^2 - 2ab*cos(gamma).
      gamma = ba.angle_rad
      length = np.sqrt(bond1.length**2 + bond2.length**2
                       - 2 * bond1.length * bond2.length * np.cos(gamma))

      # Propagation of uncertainty assuming uncorrelated errors.
      dl_outer = 0.5 / length
      dl_dgamma = (2 * bond1.length * bond2.length * np.sin(gamma)) * dl_outer
      dl_db1 = (2 * bond1.length - 2 * bond2.length * np.cos(gamma)) * dl_outer
      dl_db2 = (2 * bond2.length - 2 * bond1.length * np.cos(gamma)) * dl_outer
      stddev = np.sqrt((dl_dgamma * ba.stddev)**2 +
                       (dl_db1 * bond1.stddev)**2 +
                       (dl_db2 * bond2.stddev)**2)
      residue_virtual_bonds[resname].append(
          Bond(ba.atom1_name, ba.atom3name, length, stddev))

  return (residue_bonds,
          residue_virtual_bonds,
          residue_bond_angles)


# Between-residue bond lengths for general bonds (first element) and for Proline
# (second element).
between_res_bond_length_c_n = [1.329, 1.341]
between_res_bond_length_stddev_c_n = [0.014, 0.016]

# Between-residue cos_angles.
between_res_cos_angles_c_n_ca = [-0.5203, 0.0353]  # degrees: 121.352 +- 2.315
between_res_cos_angles_ca_c_n = [-0.4473, 0.0311]  # degrees: 116.568 +- 1.995


def make_atom14_dists_bounds(overlap_tolerance=1.5,
                             bond_length_tolerance_factor=15):
  """compute upper and lower bounds for bonds to assess violations."""
  restype_atom14_bond_lower_bound = np.zeros([21, 14, 14], np.float32)
  restype_atom14_bond_upper_bound = np.zeros([21, 14, 14], np.float32)
  restype_atom14_bond_stddev = np.zeros([21, 14, 14], np.float32)
  residue_bonds, residue_virtual_bonds, _ = load_stereo_chemical_props()
  for restype, restype_letter in enumerate(restypes):
    resname = restype_1to3[restype_letter]
    atom_list = restype_name_to_atom14_names[resname]

    # create lower and upper bounds for clashes
    for atom1_idx, atom1_name in enumerate(atom_list):
      if not atom1_name:
        continue
      atom1_radius = van_der_waals_radius[atom1_name[0]]
      for atom2_idx, atom2_name in enumerate(atom_list):
        if (not atom2_name) or atom1_idx == atom2_idx:
          continue
        atom2_radius = van_der_waals_radius[atom2_name[0]]
        lower = atom1_radius + atom2_radius - overlap_tolerance
        upper = 1e10
        restype_atom14_bond_lower_bound[restype, atom1_idx, atom2_idx] = lower
        restype_atom14_bond_lower_bound[restype, atom2_idx, atom1_idx] = lower
        restype_atom14_bond_upper_bound[restype, atom1_idx, atom2_idx] = upper
        restype_atom14_bond_upper_bound[restype, atom2_idx, atom1_idx] = upper

    # overwrite lower and upper bounds for bonds and angles
    for b in residue_bonds[resname] + residue_virtual_bonds[resname]:
      atom1_idx = atom_list.index(b.atom1_name)
      atom2_idx = atom_list.index(b.atom2_name)
      lower = b.length - bond_length_tolerance_factor * b.stddev
      upper = b.length + bond_length_tolerance_factor * b.stddev
      restype_atom14_bond_lower_bound[restype, atom1_idx, atom2_idx] = lower
      restype_atom14_bond_lower_bound[restype, atom2_idx, atom1_idx] = lower
      restype_atom14_bond_upper_bound[restype, atom1_idx, atom2_idx] = upper
      restype_atom14_bond_upper_bound[restype, atom2_idx, atom1_idx] = upper
      restype_atom14_bond_stddev[restype, atom1_idx, atom2_idx] = b.stddev
      restype_atom14_bond_stddev[restype, atom2_idx, atom1_idx] = b.stddev
  return {'lower_bound': restype_atom14_bond_lower_bound,  # shape (21,14,14)
          'upper_bound': restype_atom14_bond_upper_bound,  # shape (21,14,14)
          'stddev': restype_atom14_bond_stddev,  # shape (21,14,14)
         }
  
