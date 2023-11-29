import warnings
warnings.filterwarnings('ignore')
# We import all our dependencies.
import numpy as np
import torch
import sys
sys.path.append('../../../')
from models.lvae import LadderVAE
from lib.gaussianMixtureNoiseModel import GaussianMixtureNoiseModel
from boilerplate import boilerplate
import lib.utils as utils
import training
from tifffile import imread
from matplotlib import pyplot as plt
from tqdm import tqdm

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print("Using device: ", device)

path="/group/jug/Sheida/maester_data/download/high_c1/"
observation= imread(path+'high_c1_source.tif')

train_data = observation[:int(0.85*observation.shape[0])]
val_data= observation[int(0.85*observation.shape[0]):]
print("Shape of training images:", train_data.shape, "Shape of validation images:", val_data.shape)

patch_size = 64
img_width = observation.shape[2]
img_height = observation.shape[1]
num_patches = int(float(img_width*img_height)/float(patch_size**2)*1)
train_images = utils.extract_patches(train_data, patch_size, num_patches)
train_images = train_images[:10000] # We limit training patches to 1000 to speed up training but it is not necessary
val_images = utils.extract_patches(val_data, patch_size, num_patches)
val_images = val_images[:500] # We limit validation patches to 1000 to speed up training but it is not necessary
test_images = val_images[:100]
img_shape = (train_images.shape[1], train_images.shape[2])
print("Shape of training images:", train_images.shape, "Shape of validation images:", val_images.shape)

model_name = "convallaria"
directory_path = "./Trained_model/" 

# Data-specific
gaussian_noise_std = None
noise_model_params= np.load("./data/Convallaria_diaphragm/GMMNoiseModel_convallaria_3_2_calibration.npz")
noiseModel = None #GaussianMixtureNoiseModel(params = noise_model_params, device = device)

# Training-specific
batch_size=128
virtual_batch = 8
lr=3e-4
max_epochs = 500
steps_per_epoch=400
test_batch_size=100

# Model-specific
num_latents = 6
z_dims = [32]*int(num_latents)
blocks_per_layer = 5
batchnorm = True
free_bits = 1.0

train_loader, val_loader, test_loader, data_mean, data_std = boilerplate._make_datamanager(train_images,val_images,
                                                                                           test_images,batch_size,
                                                                                           test_batch_size)

model = LadderVAE(z_dims=z_dims,blocks_per_layer=blocks_per_layer,data_mean=data_mean,data_std=data_std,noiseModel=noiseModel,
                  device=device,batchnorm=batchnorm,free_bits=free_bits,img_shape=img_shape).cuda()

model.train() # Model set in training mode


training.train_network(model=model,lr=lr,max_epochs=max_epochs,steps_per_epoch=steps_per_epoch,directory_path=directory_path,
                       train_loader=train_loader,val_loader=val_loader,test_loader=test_loader,
                       virtual_batch=virtual_batch,gaussian_noise_std=gaussian_noise_std,
                       model_name=model_name,val_loss_patience=30)