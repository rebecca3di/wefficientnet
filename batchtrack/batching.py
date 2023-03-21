from torch.utils.data import Dataset
from PIL import Image

import torch
import numpy as np
import pickle as pk

class csi_n_pic_dataset(Dataset):
    def __init__(self, pk_path):
        self.df = pk.load(open(pk_path, 'rb'))
        self.len = len(self.df)

    def __getitem__(self, idx):
        d = dict(self.df.iloc[idx])
        return d['jpg']

    def __len__(self):
        return self.len


def collate_fn(batch, posix='.pk1'):
    jpgs = [np.array(Image.open(_)) for _ in batch]
    csiamp = [pk.load(open(_[:-4] + posix, 'rb'))[0] for _ in batch]
    csipha = [pk.load(open(_[:-4] + posix, 'rb'))[1] for _ in batch]
    return np.stack(jpgs, axis=0), (np.stack(csiamp, axis=0), np.stack(csipha, axis=0))
