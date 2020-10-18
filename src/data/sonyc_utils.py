import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, DataLoader

def get_sonyc_loaders(config):
    data_path = config['data_path']
    model = config['mode']
    coarse = config['coarse']
    data_dict = np.load(os.path.join(data_path, f"SONYC_data_MODE_{mode}.npy"), allow_pickle=True)
    data_dict = data_dict.item()

    train_dataset = SONYCDataset(data_dict, 'train', coarse)
    val_dataset = SONYCDataset(data_dict, 'val', coarse)
    test_dataset = SONYCDataset(data_dict, 'test', coarse)

    batch_size = config['hparams']['batch_size']
    train_loader = torch.utils.data.DataLoader(train_dataset, 100, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, 100)
    test_loader = torch.utils.data.DataLoader(test_dataset, 100)

    return (train_loader, val_loader, test_loader), train_dataset

class SONYCDataset(Dataset):
    def __init__(self, data_dict, split, coarse=0):
        data = data_dict[split]['coarse' if coarse else 'fine']
        self.X = data['X']/255.0
        self.Y_true = data['Y_true']
        self.Y_mask = data['Y_mask']
        self.length = self.X.shape[0]
        self.H, self.W = self.X[0].shape
        self.labelled = h['Y_mask'][:].sum()
        self.total = np.prod(h['Y_mask'].shape)
    
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        t_dtype = torch.float32
        X = self.X[index]
        Y_true = self.Y_true[index]
        Y_mask = self.Y_mask[index]
        X = torch.tensor(X, requires_grad=False, dtype=t_dtype)
        Y_true = torch.tensor(Y_true.astype(float), requires_grad=False, dtype=t_dtype)
        Y_mask = torch.BoolTensor(Y_mask.astype(bool))
        return X, Y_true, Y_mask
