import warnings
warnings.filterwarnings('ignore')
# We import all our dependencies.
import numpy as np
import torch
import sys
sys.path.append('../../../')
sys.path.append('/home/sheida.rahnamai/GIT/HDN/')
from models.lvae import LadderVAE
from boilerplate import boilerplate
import lib.utils as utils
import training
from tifffile import imread
from matplotlib import pyplot as plt
from tqdm import tqdm
import wandb
import random
import tifffile as tiff
import glob

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

path="/group/jug/Sheida/pancreatic beta cells/download/high_c1/contrastive/patches/"
# path = "/localscratch/"
patch_size = 64

paths = sorted(glob.glob(path+"training/img/*.tif"))
train_images = tiff.imread(paths)
paths = sorted(glob.glob(path+"training/mask/*.tif"))
train_y = np.array(tiff.imread(paths), dtype=np.int32)
paths = sorted(glob.glob(path+"validation/img/*.tif"))
val_images = tiff.imread(paths)
paths = sorted(glob.glob(path+"validation/mask/*.tif"))
val_y = np.array(tiff.imread(paths), dtype=np.int32)
paths = sorted(glob.glob(path+"testing/img/*.tif"))
test_images = tiff.imread(paths)
paths = sorted(glob.glob(path+"testing/mask/*.tif"))
test_y = np.array(tiff.imread(paths), dtype=np.int32)

model_name = "Contrastive_MAE"
directory_path = "/group/jug/Sheida/HVAE/101/"
# directory_path = "test"
# Data-specific
gaussian_noise_std = None
noiseModel = None 
# Training-specific
batch_size=128
# batch_size=16
virtual_batch = 64
lr=3e-4
max_epochs = 500
steps_per_epoch = 1
test_batch_size=100

# Model-specific
num_latents = 3
z_dims = [32]*int(num_latents)
blocks_per_layer = 5
mask_size = 4
batchnorm = True
free_bits = 0.0 # if KLD is less than 1 then the loss won't be calculated
contrastive_learning = True
cl_mode = 'min max'

debug             = False #[True, False]
save_output       = True #[True, False]
use_non_stochastic = True
project           = 'Contrastive_MAE'
img_shape = (64,64)

# train_loader, val_loader, test_loader, data_mean, data_std = boilerplate._make_datamanager(train_images,train_y,val_images,val_y,
train_loader, val_loader, data_mean, data_std = boilerplate._make_datamanager(train_images,train_y,val_images,val_y,
                                                                                           test_images,batch_size,
                                                                                           test_batch_size)

model = LadderVAE(z_dims=z_dims,blocks_per_layer=blocks_per_layer,data_mean=data_mean,data_std=data_std,noiseModel=noiseModel,
                  device=device,batchnorm=batchnorm,free_bits=free_bits,img_shape=img_shape,contrastive_learning=contrastive_learning,cl_mode=cl_mode,mask_size=mask_size, use_non_stochastic=use_non_stochastic, learn_top_prior=True).cuda()

model.train() # Model set in training mode

training.train_network(model=model,lr=lr,max_epochs=max_epochs,steps_per_epoch=steps_per_epoch,directory_path=directory_path,
                    #    train_loader=train_loader,val_loader=val_loader,test_loader=test_loader,
                       train_loader=train_loader,val_loader=val_loader,
                       virtual_batch=virtual_batch,gaussian_noise_std=gaussian_noise_std,
                       model_name=model_name,val_loss_patience=100, debug=debug, save_output=save_output, project_name=project, batch_size=batch_size, cl_w = 1e-1, kl_w = 1)

