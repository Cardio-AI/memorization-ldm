#Load required modules
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from monai.data import DataLoader, Dataset
from tqdm import tqdm
import json

from networks.SiameseNetwork import SiameseNetwork

from pytorch_metric_learning.distances import  CosineSimilarity
from pytorch_metric_learning.losses import NTXentLoss
from dataset.nih_chest_xray_cl import NIHXRayDataset
from torch.utils.tensorboard import SummaryWriter

def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)


class MyDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
#Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/mnt/sds/sd20i001/salman/data/NIHXRay')
parser.add_argument('--dataset', default='NIHXRay')
parser.add_argument('--n_epochs', type = int,default=200)
parser.add_argument('--batch_size', type = int, default=64)
parser.add_argument('--training_samples', type = int, default=10000)
parser.add_argument('--val_interval', type = int, default=1)
parser.add_argument('--base_lr', type = float, default=1e-4)#
parser.add_argument('--save_model_interval', type = int, default=20)
parser.add_argument('--ckpt_dir', type = str, default='ckpt/')
parser.add_argument('--details', type = str, default='')
parser.add_argument('--downsample', type = int, default=2)
parser.add_argument('--temperature',type = float, default=0.07)
parser.add_argument('--multi_gpu', action='store_true')
parser.add_argument('--single_labeled',  action='store_true')#
parser.add_argument('--exp' , default='exp_2D_1/') # Experiment name
args = parser.parse_args()

#Arguments
data_dir = args.data_dir
dataset = args.dataset
n_epochs = args.n_epochs
batch_size =  args.batch_size
training_samples =  args.training_samples
val_interval = args.val_interval
save_model_interval = args.save_model_interval
ckpt_dir = args.ckpt_dir
downsample = args.downsample
base_lr = args.base_lr
temperature = args.temperature
multi_gpu = args.multi_gpu
single_labeled = args.single_labeled
exp = args.exp
isExist = os.path.exists(ckpt_dir + exp)
if not isExist:
   # Create a new directory because it does not exist
   os.makedirs(ckpt_dir+ exp)
#Save arguments
with open(ckpt_dir + exp+'/arguments.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2) 
#Data
if 'NIHXRay' in dataset:
    train_data = NIHXRayDataset(root_dir= data_dir, split = 'train', training_samples = training_samples , donwsample = downsample, single_labeled = single_labeled)
    val_data = NIHXRayDataset(root_dir= data_dir, split = 'val', donwsample = downsample)

#Data loaders
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, persistent_workers=False)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0, persistent_workers=False)
cosine_similarity = CosineSimilarity()

#Define model
model = SiameseNetwork()
if multi_gpu: model = MyDataParallel(model)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

#Training
optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs, eta_min = base_lr/2.0)
LossTr = NTXentLoss(temperature=temperature)


writer = SummaryWriter(log_dir= ckpt_dir)
epoch_losses =[]
val_losses = []
for epoch in range(n_epochs):
    
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
    progress_bar.set_description(f"Epoch {epoch}")    

    epoch_loss = 0
    model.train()
    for train_step, batch in progress_bar:
        optimizer.zero_grad()
        Pos11 = batch['data'].to(device); Pos12 = batch['data_pos'].to(device);  

        #Obtain positive and negative embddings
        
        PosEmb11 = model(Pos11.to(device), resnet_only = True); PosEmb12 = model(Pos12.to(device), resnet_only = True)

        #print(PosEmb11.max())
        

        Labels = torch.arange(PosEmb11.shape[0])
        LossPos1 = LossTr(torch.cat((PosEmb11, PosEmb12), dim = 0), torch.cat((Labels, Labels), dim = 0))
   
        LTotal =  LossPos1  
        
        LTotal.backward()
        epoch_loss += LTotal.item()

        optimizer.step()
    scheduler.step()
    epoch_losses.append(epoch_loss / (train_step + 1))

    #Plot latent embeddings
    val_loss = 0;pos_sim=[];neg_sim=[];neg_sim_aug=[]
    model.eval()
    for val_step, batch in enumerate(val_loader):
        
        Pos11 = batch['data'].to(device); Pos12 = batch['data_pos'].to(device) 
        with torch.no_grad():
        #predictions
            PosEmb11 = model(Pos11.to(device), resnet_only = True); PosEmb12 = model(Pos12.to(device), resnet_only = True)

        #loss

        Labels = torch.arange(PosEmb11.shape[0])
        val_loss += LossTr(torch.cat((PosEmb11, PosEmb12), dim = 0),  torch.cat((Labels, Labels), dim = 0)).item()       
        similarity_pos = cosine_similarity(PosEmb11, PosEmb12).cpu().numpy()
        similarity_neg = cosine_similarity(PosEmb11, PosEmb11).cpu().numpy()

        pos_sim.append( np.diag(similarity_pos))
        neg_sim.append (similarity_neg[np.triu_indices_from(similarity_neg, k=1)])
        neg_sim_aug.append (similarity_pos[np.triu_indices_from(similarity_pos, k=1)])
        torch.cuda.empty_cache()
    val_loss /= (val_step+1)
    val_losses.append(val_loss )

    writer.add_histogram('Positive samples', np.hstack(pos_sim), epoch)
    writer.add_histogram('Negative samples', np.hstack(neg_sim), epoch)
    writer.add_histogram('Negative samples Augmented', np.hstack(neg_sim_aug), epoch)
    writer.add_scalar('Train/Loss', epoch_losses[-1], epoch)
    writer.add_scalar('Val/Loss', val_losses[-1], epoch)

    if (epoch + 1) % save_model_interval == 0 or epoch==0:  
        if multi_gpu:
            torch.save(model.module.state_dict(), ckpt_dir +"/model"+ str(epoch))
        else:
            torch.save(model.state_dict(), ckpt_dir +"/model"+ str(epoch))
    if (epoch >1) and (val_loss < min(val_losses[:-1])):
        if multi_gpu:
            torch.save(model.module.state_dict(), ckpt_dir +"model_best")
        else:
            torch.save(model.state_dict(), ckpt_dir +"model_best")
 