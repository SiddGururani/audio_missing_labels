import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, DataLoader

"""
TODO: Cleanup code for ICASSP submission
"""

def get_openmic_loaders(config):
    abs_path = config['data_path']
    full_dataset = HDF5Dataset(os.path.join(abs_path, 'openmic_train.h5'))
    train_val_split = np.load(os.path.join(abs_path, 'train_val.split'))
    train_dataset = Subset(full_dataset, train_val_split['train'])
    val_dataset = Subset(full_dataset, train_val_split['val'])
    test_dataset = HDF5Dataset(os.path.join(abs_path, 'openmic_test.h5'))

    batch_size = config['hparams']['batch_size']
    train_loader = torch.utils.data.DataLoader(train_dataset, 100, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, 100)
    test_loader = torch.utils.data.DataLoader(test_dataset, 100)
    return (train_loader, val_loader, test_loader), train_dataset

def binarize_targets(targets, threshold=0.5):
    targets[targets < threshold] = 0
    targets[targets > 0] = 1
    return targets

class HDF5Dataset(Dataset):
    def __init__(self, h5_path, half_precision=False, feat_type='vgg', specaug=False, aug=False):
        h = h5py.File(h5_path, 'r')
        self.h = None
        self.h5_path = h5_path
        self.length = h['vggish'].shape[0]
        self.H, self.W = h['vggish'][0].shape
        self.labelled = h['Y_mask'][:].sum()
        self.total = np.prod(h['Y_mask'].shape)
        # self.pos_weights = torch.load('./data/pos_weights.pth')
        self.half = half_precision
        self.feat_type = feat_type
        self.specaug = specaug
#         self.wave = np.load('waveshape.npy')
        self.Y_true = h['Y_true'][:]
        self.Y_mask = h['Y_mask'][:]
        h.close()
    
    def get_vggish(self):
        if self.h is None:
            self.h = h5py.File(self.h5_path, 'r', libver='latest', swmr=True)
            self.vggish = self.h['vggish'][:]/255.0
            self.spec = self.h['spec']
            self.audio = self.h['audio']
            self.Y_true = self.h['Y_true']
            self.Y_mask = self.h['Y_mask']
        return self.vggish

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Lazy opening of h5 file to enable multiple workers
        # in the dataloader init. Only for augmentation though
        if self.h is None:
            self.h = h5py.File(self.h5_path, 'r', libver='latest', swmr=True)
            self.vggish = self.h['vggish'][:]/255.0
            self.spec = self.h['spec']
            self.audio = self.h['audio']
            self.Y_true = self.h['Y_true'][:]
            self.Y_mask = self.h['Y_mask'][:]
        # Add routines to decide which data to pick up depending on what augmentation to use
        # In future iterations, also experiment with outputting two augmented versions of X
        t_dtype = torch.float16 if self.half else torch.float32
        if self.feat_type == 'vgg':
            X = self.vggish[index]
        elif self.feat_type == 'vgg_wave':
            i = np.random.choice(range(7))
            if i == 0:
                X = self.vggish[index]
            else:
                X = self.wave[index][i-1]
        elif self.feat_type == 'spec':
            X = self.spec[index]
#             if self.specaug:
#                 X = specaugment(X)
        elif self.feat_type == 'audio':
            X = self.audio[index]
        # X = X.reshape(self.H, self.W)
        Y_true = binarize_targets(self.Y_true[index])
        Y_mask = self.Y_mask[index]
        X = torch.tensor(X, requires_grad=False, dtype=t_dtype)
        Y_true = torch.tensor(Y_true.astype(float), requires_grad=False, dtype=t_dtype)
        Y_mask = torch.BoolTensor(Y_mask.astype(bool))
        return X, Y_true, Y_mask

    def terminate(self):
        if self.h is not None:
            self.h.close()
            
def audio_to_vggish(audio):
    feats = VGGish(audio)
    return feats/255.0