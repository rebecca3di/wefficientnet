from torch.utils.data import Dataset
from PIL import Image
# import torch

import numpy as np
import pickle as pk

class csi_n_pic_dataset(Dataset):
    def __init__(self, pk_path):
        self.df = pk.load(open(pk_path, 'rb'))
        prefix = 'D:/caizhijie/data/archive_0328/archive/'

        if prefix.startswith('D'):
            self.df['pathpack'] = self.df['pathpack'].apply(lambda x: [prefix + _[32:] for _ in x])

        self.len = len(self.df)

    def __getitem__(self, idx):
        d = dict(self.df.iloc[idx])
        return d
    
    def __len__(self):
        return self.len
    

def collate_fn(batch, posix='.pk_2_1'):
    n_steps = 10
    n_frames = 5

    # try:
    jpgs = [np.stack([np.array(Image.open(_)) for _ in b['pathpack'][:n_frames]], axis=0) for b in batch]
    csiamp = [np.concatenate([pk.load(open(_[:-4] + posix, 'rb'))[0] for _ in b['pathpack'][:n_steps]], axis=0)[:64, ...] for b in batch]
    csipha = [np.concatenate([pk.load(open(_[:-4] + posix, 'rb'))[1] for _ in b['pathpack'][:n_steps]], axis=0)[:64, ...] for b in batch]
    # except:
        # jpgs = [np.array(Image.open(prefix + _['pathpack'][0][32:])) for _ in batch]
        # csiamp = [np.concatenate([pk.load(open(prefix + _[32:-4] + posix, 'rb'))[0] for _ in b['pathpack'][:n_steps]], axis=0)[:25, ...] for b in batch]
        # csipha = [np.concatenate([pk.load(open(prefix + _[32:-4] + posix, 'rb'))[1] for _ in b['pathpack'][:n_steps]], axis=0)[:25, ...] for b in batch]

    
    return np.stack(jpgs, axis=0), (np.stack(csiamp, axis=0), np.stack(csipha, axis=0))