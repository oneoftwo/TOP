import numpy as np
import random
from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers
from rdkit.Chem.Descriptors import MaxPartialCharge, MinPartialCharge, MolWt, \
        NumValenceElectrons, NumRadicalElectrons, qed, TPSA 
from rdkit.Chem.Crippen import MolLogP, MolMR
from rdkit.Chem.Lipinski import NumRotatableBonds, NumHAcceptors, NumHDonors, \
        NumHeteroatoms, NumAliphaticCarbocycles, NumAliphaticHeterocycles, \
        NumAliphaticRings, NumAromaticCarbocycles, NumAromaticHeterocycles, \
        NumAromaticRings, NumSaturatedCarbocycles, NumSaturatedHeterocycles, \
        NumSaturatedRings


def txt_to_smiles_list(fn):
    f = open(fn, 'r')
    smiles_list = []
    for line in f:
        smiles_list.append(line.strip())
    return smiles_list


def smiles_list_to_txt(fn, smiles_list):
    f = open(fn, 'w')
    for idx, smiles in enumerate(smiles_list):
        f.write(smiles)
        if not idx == len(smiles_list) - 1:
            f.write('\n')
    return None


def sanitize_smiles(smiles, stereo=True):
    mol = Chem.MolFromSmiles(smiles)
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
    return smiles


def get_physical_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    pq = []
    pq.append(NumAliphaticCarbocycles(mol))
    pq.append(NumAliphaticHeterocycles(mol))
    pq.append(NumAliphaticRings(mol))
    pq.append(NumAromaticCarbocycles(mol))
    pq.append(NumAromaticHeterocycles(mol))
    pq.append(NumAromaticRings(mol))
    pq.append(NumSaturatedCarbocycles(mol))
    pq.append(NumSaturatedHeterocycles(mol))
    pq.append(NumSaturatedRings(mol))
    pq.append(NumHDonors(mol))
    pq.append(NumHAcceptors(mol))
    pq.append(NumRotatableBonds(mol))
    pq.append(NumHeteroatoms(mol))
    # DrugLikeness
    pq.append(MolLogP(mol))
    pq.append(MolWt(mol))
    pq.append(MolMR(mol))
    pq.append(TPSA(mol))
    pq.append(qed(mol))
    # Reactivity
    pq.append(NumValenceElectrons(mol))
    pq.append(NumRadicalElectrons(mol))
    pq = scaling(pq, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, \
            0.0, -5.9718, 52.036, 5.163, 0.0, 0.08327133915786711, 16.0, 0.0], [9.0, \
            13.0, 13.0, 2.0, 3.0, 3.0, 9.0, 13.0, 13.0, 5.0, 9.0, 9.0, 13.0, 7.4363, \
            147.952, 89.257, 158.88, 0.7146923410153729, 68.0, 5.0])
    return pq


def scaling(arr, min_val, diff):
    arr = np.array(arr)
    min_val = np.array(min_val)
    diff = np.array(diff)
    scaled_arr = (arr - min_val) / diff
    return np.clip(scaled_arr, 0, 1) 


if __name__ == '__main__':
    a = sanitize_smiles('C[C@H](O)c1ccccc1')
    print(get_physical_properties('C[C@H](O)c1ccccc1'))

