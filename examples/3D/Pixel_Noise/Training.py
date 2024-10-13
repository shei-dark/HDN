import os
import warnings
import urllib
import zipfile

warnings.filterwarnings("ignore")
# We import all our dependencies.
import numpy as np
import torch
import sys

sys.path.insert(0, "/home/sheida.rahnamai/GIT/HDN/")
from torch.utils.data import DataLoader
from models.lvae import LadderVAE
from lib.gaussianMixtureNoiseModel import GaussianMixtureNoiseModel
from boilerplate import boilerplate
from boilerplate.dataloader import (
    Custom3DDataset,
    CombinedCustom3DDataset,
    BalancedBatchSampler,
    CombinedBatchSampler,
)
import lib.utils as utils
import training
from tifffile import imread
from scipy import ndimage
from matplotlib import pyplot as plt
from tqdm import tqdm
import tifffile as tiff
from glob import glob
from itertools import chain
import pickle

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

patch_size = 64

gaussian_noise_std = None


model_name = "3D_HVAE"
directory_path = "/group/jug/Sheida/HVAE/3D/v08/"
noiseModel = None

# Training-specific
batch_size = 16
lr = 3e-4
max_epochs = 500

# Model-specific
load_checkpoint = False
num_latents = 3
z_dims = [32] * int(num_latents)
blocks_per_layer = 5
batchnorm = True
free_bits = 0.0

alpha = 1
beta = 1e-1
gamma = 1e-2
# contrastive
mask_size = 5
contrastive_learning = True
margin = 250
lambda_contrastive = 0.5
labeled_ratio = 1.0

use_wandb = True

classes = ["uncategorized", "nucleus", "granule", "mitochondria"]
train_labeled_indices = []
val_labeled_indices = []
for cls in classes:
    with open(
        f"/group/jug/Sheida/pancreatic beta cells/download/3d/train/1_percent_{cls}.pickle",
        "rb",
    ) as file:
        train_labeled_indices.extend(pickle.load(file))
    with open(
        f"/group/jug/Sheida/pancreatic beta cells/download/3d/val/1_percent_{cls}.pickle",
        "rb",
    ) as file:
        val_labeled_indices.extend(pickle.load(file))

# train data

data_dir = "/group/jug/Sheida/pancreatic beta cells/download/3d/"
keys = ["high_c1", "high_c2", "high_c3"]

# Load source images
train_img_paths = sorted(
    list(
        chain.from_iterable(
            glob(os.path.join(data_dir, "train", key, f"{key}_source_*"))
            for key in keys
        )
    )
)
train_lbl_paths = sorted(
    list(
        chain.from_iterable(
            glob(os.path.join(data_dir, "train", key, f"{key}_gt_*")) for key in keys
        )
    )
)
val_img_paths = sorted(
    list(
        chain.from_iterable(
            glob(os.path.join(data_dir, "val", key, f"{key}_source_*")) for key in keys
        )
    )
)
val_lbl_paths = sorted(
    list(
        chain.from_iterable(
            glob(os.path.join(data_dir, "val", key, f"{key}_gt_*")) for key in keys
        )
    )
)
print(
    f"Training image paths: {train_img_paths}\nTraining label paths: {train_lbl_paths}\nValidation image paths: {val_img_paths}\nValidation label paths: {val_lbl_paths}"
)

train_images = [tiff.imread(path) for path in train_img_paths]
train_labels = [tiff.imread(path) for path in train_lbl_paths]

val_images = [tiff.imread(path) for path in val_img_paths]
val_labels = [tiff.imread(path) for path in val_lbl_paths]

# compute mean and std of the data

# Flatten train_images
train_elements = np.concatenate([image.flatten() for image in train_images])
data_mean = np.mean(train_elements)
data_std = np.std(train_elements)

# normalizing the data
for idx in tqdm(range(len(train_images)), "Normalizing train data"):
    train_images[idx] = (train_images[idx] - data_mean) / data_std
for idx in tqdm(range(len(val_images)), "Normalizing validation data"):
    val_images[idx] = (val_images[idx] - data_mean) / data_std

train_set = CombinedCustom3DDataset(train_images, train_labels, train_labeled_indices)
val_set = CombinedCustom3DDataset(val_images, val_labels, val_labeled_indices)

# train_set = dataloader.Custom3DDataset(train_images, train_labels)
# val_set = dataloader.Custom3DDataset(val_images, val_labels)

train_sampler = CombinedBatchSampler(train_set, batch_size, labeled_ratio=labeled_ratio)
# train_sampler = dataloader.BalancedBatchSampler(train_set, batch_size)

train_loader = DataLoader(train_set, sampler=train_sampler)

val_sampler = CombinedBatchSampler(val_set, batch_size, labeled_ratio=labeled_ratio)
# val_sampler = dataloader.BalancedBatchSampler(val_set, batch_size)

val_loader = DataLoader(val_set, sampler=val_sampler)

img_shape = (64, 64, 64)

if load_checkpoint:
    model = torch.load("")
else:
    model = LadderVAE(
        z_dims=z_dims,
        blocks_per_layer=blocks_per_layer,
        data_mean=data_mean,
        data_std=data_std,
        noiseModel=noiseModel,
        conv_mult=3,
        device=device,
        batchnorm=batchnorm,
        free_bits=free_bits,
        img_shape=img_shape,
        grad_checkpoint=True,
        mask_size=mask_size,
        contrastive_learning=contrastive_learning,
        margin=margin,
        lambda_contrastive=lambda_contrastive,
        labeled_ratio=labeled_ratio,
    ).cuda()

model.train()  # Model set in training mode

training.train_network(
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
)
