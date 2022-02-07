import numpy as np
import time
from tqdm import tqdm
import torch
from torch import nn
import _dataset as DATASET 


def calc_precision(pred, true):
    from sklearn.metrics import precision_score
    a = precision_score(true, pred, zero_division=0)
    b = precision_score(true, pred, zero_division=1)
    if (a == 0) and (b==1):
        return -1
    else:
        return a


def calc_recall(pred, true):
    from sklearn.metrics import recall_score
    return recall_score(true, pred)


def calc_auc_roc(pred, true):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(true, pred)


def calc_acc(pred, true):
    from sklearn.metrics import accuracy_score
    return accuracy_score(true, pred)


def calc_auc_prc(pred, true):
    from sklearn.metrics import auc
    from sklearn.metrics import precision_recall_curve
    p, r, ths = precision_recall_curve(true, pred)
    asdf = auc(r, p)
    return asdf



def run_single_epoch(model, data_loader, optimizer=None):
    
    if not optimizer == None:
        model.train()
    else:
        model.eval()

    criterion = nn.BCELoss(reduction='sum')
    
    total_loss = 0
    total_pred_prob, total_pred_label, total_true_label = [], [], []
    for batch in tqdm(data_loader):
        s, l, prop = batch['s'].float(), batch['len'].long(), batch['property']           
        prop = batch['property']
        true_label = batch['label']
        
        pred_prob = model(s, l, prop)
        pred_prob = torch.sigmoid(pred_prob)

        loss = criterion(pred_prob, true_label)
        total_loss += loss.item()

        if not optimizer == None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        pred_label = ((pred_prob) > 0.5).int()
        
        total_pred_label += pred_label.tolist()
        total_pred_prob += pred_prob.tolist()
        total_true_label += true_label.int().tolist()
    
    total_pred_prob, total_pred_label, total_true_label = \
            np.array(total_pred_prob), np.array(total_pred_label), np.array(total_true_label)
    precision = calc_precision(total_pred_label, total_true_label)
    recall = calc_precision(total_pred_label, total_true_label)
    acc = calc_acc(total_pred_label, total_true_label)
    auc_roc = calc_auc_roc(total_pred_prob, total_true_label)
    auc_prc = calc_auc_roc(total_pred_prob, total_true_label)
    
    result_dict = {'precision':precision, 'recall':recall, 'accuracy':acc, 'auc_roc':auc_roc, 'auc_prc':auc_prc}
    result_dict['loss'] = total_loss / len(data_loader.dataset)
    return model, result_dict


def train_model(model, data_loader, args):
    pass
    


if __name__ == '__main__':
    import pickle
    from torch.utils.data import DataLoader
    import _model as MODEL

    c_to_i_fn = './data/c_to_i.pkl'
    c_to_i = pickle.load(open(c_to_i_fn, 'rb'))
    
    data_list_fn = './data/tox21/tox21_NR-AR_test.pkl'
    data_list = pickle.load(open(data_list_fn, 'rb'))[:100]

    train_dataset = DATASET.SmilesDataset(data_list, c_to_i)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, \
            collate_fn=DATASET.smiles_dataset_collate_fn, drop_last=False)
    
    model = MODEL.SmilesClassifier(in_dim=len(c_to_i))
    run_single_epoch(model, train_loader, optimizer=None)


