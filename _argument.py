import argparse

def get_train_args():
    parser = argparse.ArgumentParser()
    
    # required
    parser.add_argument('--train_data_fn', required=True)
    parser.add_argument('--val_data_fn', required=True)
    parser.add_argument('--c_to_i_fn', required=True)
    
    # 
    parser.add_argument('--save_dir', required=False, default=None)
    parser.add_argument('--bs', type=int, required=False, default=16)
    parser.add_argument('--lr', type=float, required=False, default=1e-4)
    parser.add_argument('--n_epoch', type=int, required=False, default=100)

    args = parser.parse_args()

    return args
