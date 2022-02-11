import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


# langugate model
class SmilesClassifier(nn.Module):
    
    def __init__(self, in_dim, hid_dim, n_layer, property_dim=20):
        super().__init__()
        
        self.n_layer = n_layer
        self.in_dim, self.hid_dim, self.property_dim = \
                in_dim, hid_dim, property_dim

        # embedding layer for sequence char
        self.embedding = nn.Linear(in_dim, hid_dim, bias=False)
        
        # bigru cell
        self.gru = nn.GRU(input_size=hid_dim, hidden_size=hid_dim, \
                num_layers=n_layer, dropout=0.5, bidirectional=True, \
                batch_first=True)
        
        # property embedding layer
        self.fc_property_embedding = nn.Sequential(
                nn.Linear(property_dim, property_dim), 
                # nn.BatchNorm1d(property_dim),
                nn.Dropout(0.5),
                nn.LeakyReLU(),
                nn.Linear(property_dim, property_dim),
                # nn.BatchNorm1d(property_dim),
                nn.Dropout(0.5),
                nn.LeakyReLU(),
                nn.Linear(property_dim, hid_dim)
                )
        
        # readout layer
        readout_dim = 3 * hid_dim
        self.fc_readout = nn.Sequential(
                nn.Linear(readout_dim, hid_dim),
                # nn.BatchNorm1d(hid_dim),
                nn.Dropout(0.5),
                nn.LeakyReLU(),
                nn.Linear(hid_dim, hid_dim),
                # nn.BatchNorm1d(hid_dim),
                nn.Dropout(0.5),
                nn.LeakyReLU(),
                nn.Linear(hid_dim, 1)
                )

    def forward(self, s, l, prop):
        
        s = self.embedding(s) # s[b l hid_dim]
        
        packed_seq = pack_padded_sequence(s, l, \
                batch_first=True, enforce_sorted=False)
        output, h = self.gru(packed_seq)
        seq, l = pad_packed_sequence(output, batch_first=True)
        
        # get the last layer's forward and backward state 
        h_forward = h[self.n_layer * 2 - 2] # [b hd] # index last
        h_backward = h[self.n_layer * 2 - 1] # [b hd] # index 0 
        
        # get property embedding
        prop = self.fc_property_embedding(prop) # [b hd]
        
        # concat and readout
        retval = torch.cat([h_forward, h_backward, prop], dim=1) # [b 3*hd]
        pred_prob = self.fc_readout(retval).squeeze(1)
        return pred_prob
        

if __name__ == '__main__':
    pass

