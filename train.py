import numpy as np
import time
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
import _dataset as DATASET 
import _util as UTIL
def tqdm(_):
    return _


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


def run_single_epoch(model, data_loader, optimizer=None, is_gpu=True):
    
    if not optimizer == None:
        model.train()
    else:
        model.eval()

    if is_gpu:
        model.cuda()

    criterion = nn.BCELoss(reduction='sum')
    
    total_loss = 0
    total_pred_prob, total_pred_label, total_true_label = [], [], []
    for batch in tqdm(data_loader):
        s, l, prop = batch['s'].float(), batch['len'].long(), batch['property']
        true_label = batch['label']

        if is_gpu:
            s = s.cuda()
            prop = prop.cuda()
        
        pred_prob = model(s, l, prop).cpu()
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
    recall = calc_recall(total_pred_label, total_true_label)
    acc = calc_acc(total_pred_label, total_true_label)
    auc_roc = calc_auc_roc(total_pred_prob, total_true_label)
    auc_prc = calc_auc_prc(total_pred_prob, total_true_label)
    
    result_dict = {'precision':precision, 'recall':recall, 'accuracy':acc, \
            'auc_roc':auc_roc, 'auc_prc':auc_prc}
    result_dict['loss'] = total_loss / len(data_loader.dataset)
    return model, result_dict


def generate_log(epoch, rd, marker):
    output = ''
    output += f'|{epoch:^8}|'
    output += f'|{rd["train_loss"]:^12.4f}|{rd["val_loss"]:^12.4f}|'
    output += f'|{rd["accuracy"]:^12.2f}|{rd["precision"]:^12.2f}|{rd["recall"]:^12.2f}|{rd["auc_roc"]:^12.4f}|{rd["auc_prc"]:^12.4f}|'
    output += marker
    return output


def train_model(model, train_loader, val_loader, args, print_log=True, save_dir=None):
    optimizer = optim.Adam(model.parameters(), args.lr)
    best_val_loss = 1e10

    if print_log:
        output = ''
        output += f'|{"epoch":^8}|'
        output += f'|{"train_loss":^12}|{"val_loss":^12}|'
        output += f'|{"accuray":^12}|{"precision":^12}|{"recall":^12}|{"auc_roc":^12}|{"auc_prc":^12}|'
        print(output)

    for epoch in range(1, args.n_epoch+1):
        model, result_dict = run_single_epoch(model, train_loader, optimizer=optimizer)
        train_loss = result_dict['loss']
        model, result_dict = run_single_epoch(model, val_loader, optimizer=optimizer)
        val_loss = result_dict['loss']
        result_dict['train_loss'], result_dict['val_loss'] = train_loss, val_loss

        marker = ''
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            marker = '*'
        output = generate_log(epoch, result_dict, marker)
        
        if print_log:
            print(output)


if __name__ == '__main__':
    import pickle
    from torch.utils.data import DataLoader
    import _model as MODEL
    import _argument as ARGUMENT

    print('\n...train.py...\n')
    
    args = ARGUMENT.get_train_args()

    c_to_i_fn = args.c_to_i_fn
    c_to_i = pickle.load(open(c_to_i_fn, 'rb'))
    
    train_data_list_fn = args.train_data_fn
    train_data_list = pickle.load(open(train_data_list_fn, 'rb'))
    train_dataset = DATASET.SmilesDataset(train_data_list, c_to_i)
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, \
            collate_fn=DATASET.smiles_dataset_collate_fn, drop_last=False)
    
    val_data_list_fn = args.val_data_fn
    val_data_list = pickle.load(open(val_data_list_fn, 'rb'))
    val_dataset = DATASET.SmilesDataset(val_data_list, c_to_i)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True, \
            collate_fn=DATASET.smiles_dataset_collate_fn, drop_last=False)
    
    print(f'train dataset: {len(train_dataset)}')
    print(f'val dataset: {len(val_dataset)}')

    model = MODEL.SmilesClassifier(in_dim=len(c_to_i)+1)
    
    print()
    print(model)
    print()
    print('...start training...')
    UTIL.set_cuda_visible_devices(1)
    print()

    train_model(model, train_loader, val_loader, args)
    
