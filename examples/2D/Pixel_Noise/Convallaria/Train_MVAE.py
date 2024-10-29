import os
import warnings

warnings.filterwarnings("ignore")
# We import all our dependencies.
import torch
import sys

sys.path.insert(0, "/home/sheida.rahnamai/GIT/HDN/")
from models.mvae import ConvMVAE
import train_mvae
from boilerplate.dataloader import (
    Custom2DDataset,
    BalancedBatchSampler,
    CombinedBatchSampler,
)
from torch.utils.data import DataLoader
import pickle
import tifffile as tiff
from tqdm import tqdm
from boilerplate import boilerplate
import numpy as np


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Set parameters
latent_dim = 4
num_components = 4  # Number of Gaussians in the mixture

# Initialize model and optimizer



# Assume `data_loader` is prepared with 64x64 grayscale image patches
num_epochs = 50




patch_size = 64

gaussian_noise_std = None


model_name = "2D_HVAE"
directory_path = "/group/jug/Sheida/HVAE/2D/mixture/"
noiseModel = None

# Training-specific
batch_size = 8
lr = 3e-4
max_epochs = 500

# Model-specific
load_checkpoint = False
checkpoint = directory_path + "model0/2D_HVAE_best_vae.net"

alpha = 1
beta = 1e-1
gamma = 1e-2
# contrastive
mask_size = 1
label_size = 1
mode = "1x1"
contrastive_learning = True
margin = 50
lambda_contrastive = 0.5

use_wandb = True

semi_supervised = False
labeled_ratio = 1

stochastic_block_type='mixture'  # 'normal' or 'mixture'
n_components=4  # Used only for Mixture block

percent_labeled = "1_percent"

train_labeled_indices = None
val_labeled_indices = None

if semi_supervised:
    labeled_ratio = 0.5
    classes = ["uncategorized", "nucleus", "granule", "mitochondria"]
    train_labeled_indices = []
    val_labeled_indices = []
    for cls in classes:
        with open(
            f"/group/jug/Sheida/pancreatic beta cells/download/2d/train/{mode}/{percent_labeled}_{cls}.pickle",
            "rb",
        ) as file:
            train_labeled_indices.extend(pickle.load(file))
        with open(
            f"/group/jug/Sheida/pancreatic beta cells/download/2d/val/{mode}/{percent_labeled}_{cls}.pickle",
            "rb",
        ) as file:
            val_labeled_indices.extend(pickle.load(file))

# train data

data_dir = "/group/jug/Sheida/pancreatic beta cells/download/2d/"
keys = ["high_c1", "high_c2", "high_c3"]

# Load source images
train_img_paths = [
    os.path.join(data_dir + "train/" + key + f"/{key}_source.tif") for key in keys
]
train_lbl_paths = [
    os.path.join(data_dir + "train/" + key + f"/{key}_gt.tif") for key in keys
]
val_img_paths = [
    os.path.join(data_dir + "val/" + key + f"/{key}_source.tif") for key in keys
]
val_lbl_paths = [
    os.path.join(data_dir + "val/" + key + f"/{key}_gt.tif") for key in keys
]

train_images = {key: tiff.imread(path) for key, path in zip(keys, train_img_paths)}
train_labels = {key: tiff.imread(path) for key, path in zip(keys, train_lbl_paths)}

val_images = {key: tiff.imread(path) for key, path in zip(keys, val_img_paths)}
val_labels = {key: tiff.imread(path) for key, path in zip(keys, val_lbl_paths)}

for key in tqdm(keys, desc="filtering out outside of the cell"):
    filtered_image, filtered_label = boilerplate._filter_slices(
        train_images[key], train_labels[key]
    )
    train_images[key] = filtered_image
    train_labels[key] = filtered_label

    filtered_image, filtered_label = boilerplate._filter_slices(
        val_images[key], val_labels[key]
    )

    val_images[key] = filtered_image
    val_labels[key] = filtered_label

# compute mean and std of the data
all_elements = np.concatenate([train_images[key].flatten() for key in keys])
data_mean = np.mean(all_elements)
data_std = np.std(all_elements)

# normalizing the data
for key in tqdm(keys, "Normalizing data"):
    train_images[key] = (train_images[key] - data_mean) / data_std
    val_images[key] = (val_images[key] - data_mean) / data_std
train_set = Custom2DDataset(
    train_images, train_labels, patch_size, mask_size, label_size, train_labeled_indices
)
val_set = Custom2DDataset(
    val_images, val_labels, patch_size, mask_size, label_size, val_labeled_indices
)

if semi_supervised:
    train_sampler = CombinedBatchSampler(
        train_set, batch_size, labeled_ratio=labeled_ratio
    )
    val_sampler = CombinedBatchSampler(val_set, batch_size, labeled_ratio=labeled_ratio)

else:
    train_sampler = BalancedBatchSampler(train_set, batch_size)
    val_sampler = BalancedBatchSampler(val_set, batch_size)


train_loader = DataLoader(train_set, sampler=train_sampler)
val_loader = DataLoader(val_set, sampler=val_sampler)

img_shape = (64, 64)

if load_checkpoint:
    model = torch.load(checkpoint)
else:
    model = ConvMVAE(latent_dim, num_components).cuda()

print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
model.train()  # Model set in training mode


train_mvae.train_network(model, train_loader, optimizer, num_epochs, num_components)
