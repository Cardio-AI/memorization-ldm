""" Taken and adapted from https://github.com/cyclomon/3dbraingen """
import numpy as np
from torch.utils.data.dataset import Dataset
import nibabel as nib
import torchio as tio
import glob2


Normalize = tio.Compose([
    tio.transforms.RescaleIntensity(out_min_max = (-1, 1))
])

ZScore = tio.Compose([
    tio.transforms.ZNormalization()
])



class SyntheticDataset(Dataset):
    def __init__(self, root_dir, Samples = 500,Transpose = False):
        self.root = root_dir
        self.dataset = glob2.glob(self.root+'/*nii*')
        self.Samples = Samples
        self.Transpose = Transpose

    def __len__(self):
        return self.Samples

    def __getitem__(self, index, norm = 0):
        Img = nib.load(self.dataset[index])
        Img = np.asanyarray(Img.dataobj); Img = np.expand_dims(Img, axis=0)    
        if norm ==0:
            Img = Normalize(Img)
        elif norm == 1:
            Img = ZScore(Img)
        if self.Transpose:
            Img = np.transpose(Img, axes=(0,3,1,2))

        return Img
