import numpy as np 
import pandas as pd 
import pickle
import _util as UTIL
from rdkit import Chem
from tqdm import tqdm
import pickle


""" 
AR, AhR, AR-LBD, ER, ER-LBD, aromatase, PPAR-gamma, ARE, ATAD5, HSE, MMP, p53
1: toxic (positive), 0: non-toxic (negative), 1 is minor set, 0 major
"""


global tox_key_list
tox_key_list = ['NR-AR', 'NR-AhR', 'NR-AR-LBD', 'NR-ER-LBD', 'NR-Aromatase', \
        'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']


def get_mol_feature(mol):
    global tox_key_list
    sample = {}
    try:
        smiles = Chem.MolToSmiles(mol)
        sample = {}
        for tox_key in tox_key_list:
            try:
                label = mol.GetProp(tox_key)
                sample[tox_key] = int(label)
            except:
                sample[tox_key] = -1
        sample['smiles'] = UTIL.sanitize_smiles(smiles)
        sample['property'] = UTIL.get_physical_properties(smiles)
    except:
        sample = None
    return sample
        

def preprocess_tox21(fn):
    global tox_key_list
    suppl = Chem.SDMolSupplier(fn)
    data_list = []
    for mol in suppl:
        sample = get_mol_feature(mol)
        if not sample == None:
            data_list.append(sample)
    print(len(data_list))
    to_save_dict = {}
    for tox_key in tox_key_list:
        to_save = []
        for data in tqdm(data_list):
            sample = {}
            if not data[tox_key] == -1:
                sample['label'] = data[tox_key]
                sample['smiles'] = data['smiles']
                sample['property'] = data['property']
                to_save.append(sample)
        print(f'{tox_key}: {len(to_save)}')
        pickle.dump(to_save, open(f'./tox21/tox21_{tox_key}.pkl', 'wb'))
    return data_list


if __name__ == '__main__':
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    fn = './tox21/tox21_all.sdf'
    preprocess_tox21(fn)

