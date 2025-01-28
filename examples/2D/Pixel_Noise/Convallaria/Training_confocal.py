import os
import warnings
warnings.filterwarnings("ignore")
# We import all our dependencies.
import numpy as np
import torch
import sys
sys.path.insert(0, "/home/sheida.rahnamai/GIT/HDN/")
from torch.utils.data import DataLoader
from boilerplate import boilerplate
from models.lvae import LadderVAE
from boilerplate.dataloader import (
    Custom2DDataset,
    DynamicSampler
)
import training
from tqdm import tqdm
import tifffile as tiff

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

patch_size = 64

gaussian_noise_std = None


model_name = "confocal"
directory_path = "/group/jug/Sheida/HVAE/gmvae/confocal_mlp_semi/"
noiseModel = None

# Training-specific
batch_size = 512
lr = 3e-5
max_epochs = 200

# Model-specific
load_checkpoint = False
checkpoint = "/group/jug/Sheida/HVAE/gmvae/confocal_mlp/model/confocal_best_vae.net"
num_latents = 3
z_dims = [32] * int(num_latents)
blocks_per_layer = 5
batchnorm = True
free_bits = 0.0
alpha = 1
beta = 1e-4
gamma = 1e-1
# contrastive
mask_size = 1
label_size = 1
mode = "1x1"
contrastive_learning = True
margin = 50
lambda_contrastive = 0.5

use_wandb = False

# (supervised, ratio 1), (unsupervised, ratio 0), (mixed, ratio 0.25)
mode = 'supervised'
ratio = 1

stochastic_block_type = "mixture"  # 'normal' or 'mixture'
n_components = 4  # Used only for Mixture block

percent_labeled = "10_percent"

# train data

data_dir = "/group/jug/Sheida/Microsim/macrophage/"
key = "simulated_jrc_macrophage-2"

img = tiff.imread(data_dir + key + "_source.tif")
lbl = tiff.imread(data_dir + key + "_gt.tif")

num_stacks = img.shape[1]
indices = np.arange(num_stacks)
np.random.seed(42)  # For reproducibility
np.random.shuffle(indices)
# Calculate split sizes
train_size = int(0.7 * num_stacks)  # 70%
val_size = int(0.15 * num_stacks)   # 15%
test_size = num_stacks - train_size - val_size  # Remaining

# Split indices
train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]
test_indices = indices[train_size + val_size:]

# Split the stack along the second dimension (stacks)
train_images = img[:, train_indices, :, :]
val_images = img[:, val_indices, :, :]
test_images = img[:, test_indices, :, :]
train_labels = lbl[:, train_indices, :, :]
val_labels = lbl[:, val_indices, :, :]
test_labels = lbl[:, test_indices, :, :]

# compute mean and std of the data
all_elements = train_images.flatten()
data_mean = np.mean(all_elements)
data_std = np.std(all_elements)

train_stride = 10
val_stride = 10

# normalizing the data
train_images = (train_images - data_mean) / data_std
val_images = (val_images - data_mean) / data_std

train_set = Custom2DDataset(
    train_images,
    train_labels,
    patch_size,
    mask_size,
    label_size,
    train_stride,
    mode,
)
val_set = Custom2DDataset(
    val_images,
    val_labels,
    patch_size,
    mask_size,
    label_size,
    val_stride,
    mode,
)

train_sampler = DynamicSampler(train_set, batch_size)
val_sampler = DynamicSampler(val_set, batch_size)


train_loader = DataLoader(train_set, sampler=train_sampler)
val_loader = DataLoader(val_set, sampler=val_sampler)

img_shape = (64, 64)

if load_checkpoint:
    model = torch.load(checkpoint)
    model.labeled_ratio=ratio

else:
    model = LadderVAE(
        z_dims=z_dims,
        blocks_per_layer=blocks_per_layer,
        data_mean=data_mean,
        data_std=data_std,
        color_ch=3,
        noiseModel=noiseModel,
        conv_mult=2,
        device=device,
        batchnorm=batchnorm,
        free_bits=free_bits,
        img_shape=img_shape,
        grad_checkpoint=True,
        mask_size=mask_size,
        contrastive_learning=contrastive_learning,
        margin=margin,
        lambda_contrastive=lambda_contrastive,
        labeled_ratio=ratio,
        stochastic_block_type=stochastic_block_type,
        n_components=n_components,
    ).cuda()
print(model)
model.train()  # Model set in training mode

val_cl = training.train_network(
    model=model,
    lr=lr,
    max_epochs=max_epochs,
    directory_path=directory_path,
    batch_size=batch_size,
    alpha=alpha,
    beta=beta,
    gamma=gamma,
    train_loader=train_loader,
    val_loader=val_loader,
    gaussian_noise_std=gaussian_noise_std,
    model_name=model_name,
    nrows=2,
    gradient_scale=256,
    use_wandb=use_wandb,
    max_grad_norm=1,
    trial=None,
)
