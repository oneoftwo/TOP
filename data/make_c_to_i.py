import numpy as np 
import os 
import sys 
import pickle


"""   
make c_to_i from given lists of data.pkl files
"""

def get_c_to_i(smiles_list, n_hop=8):
    """   
    get c_to_i for n hop sliding window
    """
    c_to_i = {}
    for smiles in smiles_list:
        frag_list = slide_string(smiles, n_hop)
        for frag in frag_list:
            if not frag in c_to_i:
                c_to_i[frag] = len(c_to_i)
    return c_to_i


def slide_string(smiles, n_hop):
    frag_list = []
    for i in range(len(smiles) - n_hop + 1):
        frag = smiles[i:i+n_hop]
        frag_list.append(frag)
    return frag_list


def main():
    total_smiles_list = []
    for fn in sys.argv[1:]:
        data_list = pickle.load(open(fn, 'rb'))
        total_smiles_list += [x['smiles'] for x in data_list]
    c_to_i = get_c_to_i(total_smiles_list, n_hop=1)
    print(c_to_i)
    print(len(c_to_i))
    pickle.dump(c_to_i, open('./c_to_i.pkl', 'wb'))

if __name__ == '__main__':
    main()

