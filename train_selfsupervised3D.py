from vq_gan_3d.model.vqgan import Encoder, SamePadConv3d

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


from dataset.mrnet_contrastive_pairs import MRNetDatasetContrastivePairs
import torch.nn.functional as F
from scipy.spatial.distance import cdist
#from torchmetrics.functional import pairwise_cosine_similarity
from torchmetrics.utilities.compute import _safe_matmul
import argparse
import os
import json
from monai.data import DataLoader, Dataset
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm
from pytorch_metric_learning.losses import NTXentLoss
from torch.utils.tensorboard import SummaryWriter
#Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--exp' , default='exp_1') # Experiment name
parser.add_argument('--epochs' , type = int,default=200) # Trainign epochs
parser.add_argument('--batch_size', type = int, default=10) # batch size
parser.add_argument('--lr', type = float,default=0.0001)
parser.add_argument('--results_folder', default="results/")
parser.add_argument('--ckpt_file', default='') # Saved chekpoint file to continue training
parser.add_argument('--exp_details', default="First trail") # Further experiment detials to be saved in a text file
parser.add_argument('--data_directory', default="data/") # directory where data is located
parser.add_argument('--conv_only', action='store_true') # If set to true it doesn't use the dense layer at the end and only extracts conv features
parser.add_argument('--data', type = str,default='mrnet', help ='which dataset' ) # which dataset, custom can be defined
args = parser.parse_args()



epochs = args.epochs
batch_size = args.batch_size
exp = args.exp
data = args.data
lr = args.lr
ckpt_file = args.ckpt_file
results_folder = args.results_folder
exp_details = args.exp_details
data_directory = args.data_directory
conv_only = args.conv_only
sup_training = args.sup_training



isExist = os.path.exists(results_folder+exp)
if not isExist:
   # Create a new directory because it does not exist
   os.makedirs(results_folder+exp)

with open(results_folder+exp+'/arguments.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2)   
writer = SummaryWriter(log_dir= results_folder+exp)

#load dataset
if data =='mrnet':
    dataset = MRNetDatasetContrastivePairs(data_directory , task = 'acl', plane = 'sagittal',split='train')
    dataset_val = MRNetDatasetContrastivePairs(data_directory , task = 'acl', plane = 'sagittal',split='valid')

#Data loaders
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, persistent_workers=False)
val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=0, persistent_workers=False)
class Model(nn.Module):
    def __init__(self, conv_only = False,data='mrnet'):
        super().__init__()
        self.conv_only = conv_only  #If dense layer
        self.data= data
        if self.data=='pccta':
            self.downsample= [16, 16, 16] # how many times should we downssample
        elif self.data=='mrnet':
            self.downsample= [8, 64, 64] #downsampling is done to make the x,y,x dim as 4,4,4, e.g. MRNet dims are 32x256x256, Dividing by 8 64 64 makes the dims 4x4x4
        self.n_hiddens= 16 # number of hidden units
        self.image_channel = 1
        self.embedding_dim=8 
        self.encoder = Encoder(downsample = self.downsample, n_hiddens=self.n_hiddens, image_channel=self.image_channel)
        self.enc_out_ch = self.encoder.out_channels

        self.conv3D = SamePadConv3d(self.enc_out_ch, self.embedding_dim, 1)

        self.dense = nn.Sequential(nn.Flatten(),
                                nn.Linear(self.embedding_dim * 4**3, 128),
                                nn.ReLU(inplace=True),
                                nn.Dropout(0.5),
                                nn.Linear(128,32))

        self.sup_layer = nn.Sequential(nn.ReLU(inplace=True),
                            nn.Linear(32,1))

    def forward(self, x):
        x = self.encoder(x)
        h = self.conv3D(x)
        if not(self.conv_only):
            h = self.dense(h)
            fin = None
        return h, fin

model = Model(conv_only = conv_only,  data= data )
if ckpt_file:
    model.load_state_dict(torch.load(ckpt_file))
    model.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

#Training
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min = lr/2.0)
LossTr = NTXentLoss()


LossStats = np.zeros(( epochs, int(np.ceil(dataset.__len__()/batch_size))))
LossVal = np.zeros(( epochs, int(np.ceil(dataset_val.__len__()/batch_size))))
for Epoch in range(epochs):
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
    progress_bar.set_description(f"Epoch {Epoch}")       


    model.train()
    for BatchNo, Batch in progress_bar:
        optimizer.zero_grad()

        #Pos11, Pos12, Neg= BatchSample(dataset,  Batch)
        Pos11 = Batch['data1']; Pos12 = Batch['data2']
        #Obtain positive and negative embddings

        PosEmb11, _ = model(Pos11.to(device)); PosEmb12, Pred = model(Pos12.to(device))
        Labels = torch.arange(PosEmb11.shape[0])
        LossPos1 = LossTr(torch.cat((PosEmb11, PosEmb12), dim = 0), torch.cat((Labels, Labels), dim = 0))
   
        LTotal = LossPos1  
        
        LTotal.backward()
        LossStats [ Epoch, BatchNo] = LossPos1.item()
        optimizer.step()
    scheduler.step()
    #Plot latent embeddings
    model.eval()
    for val_step, Batch in enumerate(val_loader):
        
        Pos11 = Batch['data1']; Pos12 = Batch['data2']

        with torch.no_grad():
        #predictions
            PosEmb11, _ = model(Pos11.to(device)); PosEmb12, _ = model(Pos12.to(device))
        #loss
        
        Labels = torch.arange(PosEmb11.shape[0])
        LossVal [ Epoch, val_step] = LossTr(torch.cat((PosEmb11, PosEmb12), dim = 0),torch.cat((Labels, Labels), dim = 0)).item()       
        torch.cuda.empty_cache()

    torch.save(model.state_dict(), results_folder+exp+'/model_'+ str(Epoch))
    writer.add_scalar('Train/Loss', LossStats[ Epoch, :].mean(), Epoch)
    writer.add_scalar('Val/Loss', LossVal[ Epoch, :].mean(), Epoch)



        