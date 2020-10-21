import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, DataLoader

def get_sonyc_loaders(config):
    data_path = config['data_path']
    mode = config['mode']
    coarse = config['coarse']
    # data_dict = np.load(os.path.join(data_path, f"SONYC_data_orig_fixed.npy"), allow_pickle=True)
    data_dict = np.load(os.path.join(data_path, f"SONYC_OpenL3.npy"), allow_pickle=True)
    data_dict = data_dict.item()

    train_dataset = SONYCDataset(data_dict, 'train', mode, coarse)
    val_dataset = SONYCDataset(data_dict, 'val', coarse)
    test_dataset = SONYCDataset(data_dict, 'test', coarse)

    # Standardize data
    train_mean = train_dataset.X.mean(axis=(0, 1))
    train_std = np.std(train_dataset.X, axis=(0, 1))

    train_dataset.X = (train_dataset.X - train_mean)/train_std
    val_dataset.X = (val_dataset.X - train_mean)/train_std
    test_dataset.X = (test_dataset.X - train_mean)/train_std

    batch_size = config['hparams']['batch_size']
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size)

    return (train_loader, val_loader, test_loader), train_dataset

class SONYCDataset(Dataset):
    def __init__(self, data_dict, split, mode=0, coarse=0):
        labels = data_dict[split]['coarse' if coarse else 'fine']
        self.X = data_dict[split]['X']
        self.Y_true = labels['Y_true']
        self.Y_mask = labels['Y_mask']
        # if mode == 1:
        #     self.Y_mask = np.ones_like(self.Y_mask)
        self.length = self.X.shape[0]
        self.H, self.W = self.X[0].shape
        # self.labelled = h['Y_mask'][:].sum()
        # self.total = np.prod(h['Y_mask'].shape)
    
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
