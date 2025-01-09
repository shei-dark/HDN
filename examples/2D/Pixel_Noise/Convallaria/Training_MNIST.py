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
    BalancedBatchSampler,
    CombinedBatchSampler,
    MnistDataloader,
    CustomMnistDataset
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
from os.path  import join


# import optuna

input_path = '/group/jug/Sheida/archive/'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

checkpoint = ''

scale = 8
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

patch_size = 28

gaussian_noise_std = None


model_name = "EXTAC"
directory_path = "/group/jug/Sheida/HVAE/MNIST/mnist/"
noiseModel = None

# Training-specific
batch_size = 512
lr = 3e-5
max_epochs = 100

# Model-specific
load_checkpoint = False
# checkpoint = "/group/jug/Sheida/HVAE/TAC/model/best_vae.net"
num_latents = 2
z_dims = [8] * int(num_latents)
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

semi_supervised = False
labeled_ratio = 1

stochastic_block_type = "mixture"  # 'normal' or 'mixture'
n_components = 10  # Used only for Mixture block

percent_labeled = "10_percent"

train_labeled_indices = None
val_labeled_indices = None

if semi_supervised:
    labeled_ratio = 0.5


# compute mean and std of the data
data_mean = np.mean(x_train)
data_std = np.std(x_train)

x_train = (x_train - data_mean) / data_std

train_images = x_train[:int(0.8 * len(x_train))]
train_labels = y_train[:int(0.8 * len(y_train))]
val_images = x_train[int(0.8 * len(x_train)):]
val_labels = y_train[int(0.8 * len(y_train)):]

    
train_set = CustomMnistDataset(
    train_images,
    train_labels,
    patch_size,
    mask_size,
    semi_supervised,
    train_labeled_indices,
)
val_set = CustomMnistDataset(
    val_images,
    val_labels,
    patch_size,
    mask_size,
    semi_supervised,
    val_labeled_indices,
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

img_shape = (28, 28)

if load_checkpoint:
    model = torch.load(checkpoint)
else:
    model = LadderVAE(
        z_dims=z_dims,
        blocks_per_layer=blocks_per_layer,
        data_mean=data_mean,
        data_std=data_std,
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
        labeled_ratio=labeled_ratio,
        stochastic_block_type=stochastic_block_type,
        n_components=n_components,
        scale=scale,
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
