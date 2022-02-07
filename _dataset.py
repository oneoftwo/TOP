from torch.utils.data import Dataset
import numpy as np
import torch
import random
from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from tqdm import tqdm
import pickle
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SmilesDataset(Dataset):
    """ 
    smiles dataset class 
    arg: 
        data_list: list of parsed data
        c_to_i: character to index
    """
    def __init__(self, data_list, c_to_i):
        self.c_to_i = c_to_i 
        assert not 'Q' in self.c_to_i, 'Q (stop token) should not be in c_to_i'
        self.c_to_i['Q'] = len(c_to_i)
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        # smiles
        smiles = data['smiles']
        s = 'Q' + smiles + 'Q'
        s = [self.c_to_i[c] for c in s]
        s = torch.tensor(s)
        s = F.one_hot(s, num_classes=len(self.c_to_i))
        length = len(s)
        # sample
        sample = {}
        sample['label'] = data['label']
        sample['smiles'] = data['smiles']
        sample['s'] = s
        sample['property'] = data['property']
        sample['len'] = length
        sample['n_char'] = len(self.c_to_i)
        return sample
    

def smiles_dataset_collate_fn(batch):
    sample = {}
    n_char = batch[0]['n_char']
    s = torch.nn.utils.rnn.pad_sequence([b['s'] for b in batch], batch_first=True, padding_value=0)
    l =  torch.Tensor([x['len'] for x in batch])
    prop = torch.Tensor([x['property'] for x in batch])
    smiles = [x['smiles'] for x in batch]
    label = torch.Tensor([x['label'] for x in batch])
    # sample
    sample['smiles'] = smiles
    sample['s'] = s
    sample['len'] = l.long() 
    sample['property'] = prop
    sample['label'] = label
    return sample


if __name__ == '__main__':
    data_list = pickle.load(open('./data/tox21/tox21_NR-AR.pkl', 'rb'))
    c_to_i = pickle.load(open('./data/c_to_i.pkl', 'rb'))
    d = SmilesDataset(data_list, c_to_i)
    print(d[2]['n_char'])
    s = smiles_dataset_collate_fn([d[1], d[2], d[3]])
    print(s['s'])

