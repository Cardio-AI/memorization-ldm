import glob2
import torchio as tio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision.io import read_image
from scipy import signal
import random
import math
import argparse
import torch

PREPROCESSING_TRANSORMS = tio.Compose([
    tio.RescaleIntensity(out_min_max=(0, 1)),
    tio.CropOrPad(target_shape=( 1, 1024, 1024)),
])


TRAIN_TRANSFORMS = tio.Compose([
    tio.RandomAffine( degrees=(-5,5,0,0,0,0), scales = 0, default_pad_value = 'minimum',p =0.5),
    tio.RandomFlip(axes=(2), flip_probability=0.5)
    #tio.RandomGamma(log_gamma=(-0.3, 0.3))
])


VAL_TRANSFORMS = None


class NIHXRayDataset(Dataset):
    def __init__(self, root_dir,  split='train', training_samples = 1000, validation_samples = 200, augmentation = False, donwsample = 2, single_labeled = False):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.training_samples = training_samples
        self.validation_samples = validation_samples
        self.preprocessing = PREPROCESSING_TRANSORMS
        self.transforms = TRAIN_TRANSFORMS 
        self.donwsample_transform = tio.Resize((1, 1024//donwsample, 1024//donwsample))
        self.augmentation = augmentation
        self.downsample = donwsample
        self.single_labeled = single_labeled
        self.sing_idx = self._get_single_labeled()        
        self.paths = self._get_file_paths()
        self.labels, self.idxs = self._get_labels()

    def _get_single_labeled(self):
        labels_df = pd.read_csv(self.root_dir + '/Data_Entry_2017.csv')
        labels = labels_df['Finding Labels'].str.get_dummies(sep='|').to_numpy()
        sub_idx = np.where(labels.sum(axis=1)==1)[0]
        return sub_idx

    def _get_file_paths(self):
        file_paths = glob2.glob(self.root_dir + '/*/*/*.png*')
        file_paths.sort()
        if self.single_labeled:
            file_paths = [ file_paths[ii] for ii in self.sing_idx]        
        file_paths = file_paths[0:self.training_samples] if (self.split == 'train') else file_paths[-self.validation_samples:] 
        return file_paths

    def _get_labels(self):
        labels_df = pd.read_csv(self.root_dir + '/Data_Entry_2017.csv')
        if self.single_labeled:
            labels_df = labels_df.iloc[self.sing_idx]        
        labels = labels_df['Finding Labels'].str.get_dummies(sep='|')
        labels = labels.iloc[0:self.training_samples] if (self.split == 'train') else labels.iloc[ labels.index[-self.validation_samples:]]
        idxs = labels_df['Image Index']
        idxs = idxs.iloc[0:self.training_samples] if (self.split == 'train') else idxs.iloc[ idxs.index[-self.validation_samples:]]
 
        return labels, idxs

    def __len__(self):
        return len(self.paths)


    def __getitem__(self, index: int):
        img = read_image(self.paths[index]).unsqueeze(dim=1)
        img = self.preprocessing(img)
        if self.downsample>1:
            img = self.donwsample_transform(img) 
        #obtain a positive sample         
        img_pos = self.transforms(img)         
        #index of negative sample
        index_neg_all  = np.setdiff1d(range(self.__len__()),index)
        index_neg = np.random.choice(index_neg_all)
        #obtain a negative sample
        img_neg = read_image(self.paths[index_neg]).unsqueeze(dim=1)
        img_neg = self.preprocessing(img_neg)
        if self.downsample>1:
            img_neg = self.donwsample_transform(img_neg) 
        #label
        label = torch.tensor(self.labels.iloc[index,:])    ; label = torch.nonzero(label)
        if label.shape[0]>1: label = label[np.random.choice(label.shape[0])] # randomly sample an index more than one labels
        else: label = label[0] 
        img_id =  self.idxs.iloc[index]    
        return {'data': img[[0],0,:], 'data_pos': img_pos[[0],0,:], 'data_neg': img_neg[[0],0,:],  'cond': label, 'path': self.paths[index], 'img_id': img_id }
