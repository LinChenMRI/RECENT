import os
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
import numpy as np


class PairedImageInferDataset(Dataset):
    def __init__(self, moving_dir):
        self.moving_dir = moving_dir
        self.filenames = [f for f in os.listdir(moving_dir) if os.path.isfile(os.path.join(moving_dir, f))]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        moving_dir = self.moving_dir.replace('/', '\\')
        path = os.path.join(moving_dir, self.filenames[idx])
        print(path)
        data = loadmat(path)
        moving_image = data['moving_image'].astype(np.float32)
        fixed_image = data['fixed_image'].astype(np.float32)
        target = data['target'].astype(np.float32)


        moving_image = torch.tensor(moving_image, dtype=torch.float32).unsqueeze(0)
        fixed_image = torch.tensor(fixed_image, dtype=torch.float32).unsqueeze(0)
        target = torch.tensor(target, dtype=torch.float32).unsqueeze(0)

        return moving_image, fixed_image, target




