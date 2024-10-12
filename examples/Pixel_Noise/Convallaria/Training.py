# We import all our dependencies.
import sys
sys.path.append('../../../')
sys.path.append('/home/sheida.rahnamai/GIT/HDN/')
import torch
import pickle
import numpy as np
from tqdm import tqdm
from lib.dataloader import CustomDataset, CombinedCustomDataset, BalancedBatchSampler, CombinedBatchSampler
from torch.utils.data import DataLoader
import tifffile as tiff
from sklearn.utils import shuffle
import os
from boilerplate import boilerplate
from models.lvae import LadderVAE
import training

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

patch_size = 64
mask_size = 5
label_size = 5
# sample_size = 350
centre_size = 5
n_channel = 32
hierarchy_level = 3
pad_size = (patch_size - centre_size) // 2


# classes = ['uncategorized', 'nucleus', 'granule', 'mitochondria']
# train_labeled_indices = []
# val_labeled_indices = []
# for cls in classes:
#     with open(f'/group/jug/Sheida/pancreatic beta cells/download/train/0.1_percent_{cls}.pickle', 'rb') as file:
#         train_labeled_indices.extend(pickle.load(file))
#     with open(f'/group/jug/Sheida/pancreatic beta cells/download/val/0.1_percent_{cls}.pickle', 'rb') as file:
#         val_labeled_indices.extend(pickle.load(file))

# train data

data_dir = "/group/jug/Sheida/pancreatic beta cells/download/"
keys = ['high_c1', 'high_c2', 'high_c3']

# Load source images
train_img_paths = [os.path.join(data_dir + 'train/' + key + f"/{key}_source.tif") for key in keys]
train_lbl_paths = [os.path.join(data_dir + 'train/' + key + f"/{key}_gt.tif") for key in keys]
val_img_paths = [os.path.join(data_dir + 'val/' + key + f"/{key}_source.tif") for key in keys]
val_lbl_paths = [os.path.join(data_dir + 'val/' + key + f"/{key}_gt.tif") for key in keys]

train_images = {key: tiff.imread(path) for key, path in zip(keys, train_img_paths)}
train_labels = {key: tiff.imread(path) for key, path in zip(keys, train_lbl_paths)}

val_images = {key: tiff.imread(path) for key, path in zip(keys, val_img_paths)}
val_labels = {key: tiff.imread(path) for key, path in zip(keys, val_lbl_paths)}

for key in tqdm(keys, desc='filtering out outside of the cell'):
   filtered_image, filtered_label = boilerplate._filter_slices(train_images[key], train_labels[key])
   train_images[key] = filtered_image
   train_labels[key] = filtered_label

   filtered_image, filtered_label = boilerplate._filter_slices(val_images[key], val_labels[key])
   
   val_images[key] = filtered_image
   val_labels[key] = filtered_label

# compute mean and std of the data
all_elements = np.concatenate([train_images[key].flatten() for key in keys])
data_mean = np.mean(all_elements)
data_std = np.std(all_elements)

# normalizing the data
for key in tqdm(keys, 'Normalizing data'):
   train_images[key] = (train_images[key] - data_mean) / data_std
   val_images[key] = (val_images[key] - data_mean) / data_std

# train_set = CombinedCustomDataset(train_images, train_labels, train_labeled_indices)
# val_set = CombinedCustomDataset(val_images, val_labels, val_labeled_indices)
train_set = CustomDataset(train_images, train_labels, label_size=label_size)
val_set = CustomDataset(val_images, val_labels, label_size=label_size)

# One_test_image = ['high_c4']

# # Load test image
# test_img_path = os.path.join(data_dir, One_test_image[0], f"{One_test_image[0]}_source.tif")
# test_images = tiff.imread(test_img_path)

model_name = "HVAE"
directory_path = "/group/jug/Sheida/HVAE/2D/5x5_full"
# directory_path = "./test/"
# Data-specific
gaussian_noise_std = None
noiseModel = None 
# Training-specific
# batch_size=128
batch_size=8
virtual_batch = 64
lr=3e-4
max_epochs = 500
steps_per_epoch = 2
# test_batch_size=100

# Model-specific
num_latents = 3
z_dims = [32]*int(num_latents)
blocks_per_layer = 5
margin = 50
beta = 0.5
batchnorm = True
free_bits = 0.0 # if KLD is less than 1 then the loss won't be calculated
contrastive_learning = True

learn_top_prior = True

debug             = False #[True, False]
save_output       = True #[True, False]
use_non_stochastic = False
project           = '2D_HVAE'
img_shape = (64,64)
labeled_ratio = 0.25

# train_sampler = CombinedBatchSampler(train_set, batch_size, labeled_ratio=labeled_ratio)
train_sampler = BalancedBatchSampler(train_set, batch_size)
train_loader = DataLoader(train_set, sampler=train_sampler)
# val_sampler = CombinedBatchSampler(val_set, batch_size, labeled_ratio=labeled_ratio)
val_sampler = BalancedBatchSampler(val_set, batch_size)
val_loader = DataLoader(val_set, sampler=val_sampler)
    
test_set = None

feature_dim = 96

# val loader -> batch of patches of images with shape (1, batch_size, channel, patch_size, patch_size)
# then we have the centre label of the patch and the whole label of the patch
# the centre label is used for contrastive learning
# centre label of shape (1, batch_size)
# label of the patch of shape (1, batch_size, channel, patch_size, patch_size)

model = LadderVAE(z_dims=z_dims,blocks_per_layer=blocks_per_layer,data_mean=data_mean,data_std=data_std,noiseModel=noiseModel,
                  device=device,batchnorm=batchnorm,free_bits=free_bits,img_shape=img_shape,contrastive_learning=contrastive_learning,cl_mode=cl_mode,mask_size=mask_size, use_non_stochastic=use_non_stochastic, learn_top_prior=learn_top_prior, margin=margin, beta=beta, labeled_ratio=labeled_ratio).cuda()

model.train() # Model set in training mode

training.train_network(model=model,lr=lr,max_epochs=max_epochs,steps_per_epoch=steps_per_epoch,directory_path=directory_path,
                       train_loader=train_loader,val_loader=val_loader, test_set=test_set,
                       virtual_batch=virtual_batch,gaussian_noise_std=gaussian_noise_std,
                       model_name=model_name,val_loss_patience=100, debug=debug, save_output=save_output, project_name=project, batch_size=batch_size, cl_w = 1e-2, kl_w = 1e-1)

