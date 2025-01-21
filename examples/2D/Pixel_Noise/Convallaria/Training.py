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
from boilerplate.dataloader import Custom2DDataset, DynamicSampler
import training
from tqdm import tqdm
import tifffile as tiff

scale = 8
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

patch_size = 64

gaussian_noise_std = None


model_name = "epsilon_seg"
directory_path = "/group/jug/Sheida/HVAE/gmvae/orientation_invariant/"
noiseModel = None

# Training-specific
batch_size = 512
lr = 3e-5
max_epochs = 100

# Model-specific
load_checkpoint = False
checkpoint = "/group/jug/Sheida/HVAE/gmvae/decreasing_label_size/model/epsilon_seg_best_vae.net"
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

use_wandb = True

# (supervised, ratio 1), (unsupervised, ratio 0), (mixed, ratio 0.25)
mode = "supervised"
ratio = 1

stochastic_block_type = "mixture"  # 'normal' or 'mixture'
n_components = 4  # Used only for Mixture block

percent_labeled = "10_percent"

# train data

data_dir = "/group/jug/Sheida/pancreatic beta cells/download/"
keys = ["high_c1", "high_c2", "high_c3"]

img_paths = [os.path.join(data_dir + key + f"/{key}_source.tif") for key in keys]
lbl_paths = [os.path.join(data_dir + key + f"/{key}_gt.tif") for key in keys]
imgs = {key: tiff.imread(path) for key, path in zip(keys, img_paths)}
lbls = {key: tiff.imread(path) for key, path in zip(keys, lbl_paths)}
train_images, val_images, train_labels, val_labels = {}, {}, {}, {}

for key in keys:
    train_images[key] = imgs[key][np.arange(0, int(0.8 * imgs[key].shape[0]))]
    val_images[key] = imgs[key][
        np.arange(int(0.8 * imgs[key].shape[0]), imgs[key].shape[0])
    ]
    train_labels[key] = lbls[key][np.arange(0, int(0.8 * imgs[key].shape[0]))]
    val_labels[key] = lbls[key][
        np.arange(int(0.8 * imgs[key].shape[0]), imgs[key].shape[0])
    ]

valid_train = {}
valid_val = {}

for key in tqdm(keys, desc="filtering out outside of the cell"):
    filtered_image, filtered_label, valid_train[key] = boilerplate._filter_slices(
        train_images[key], train_labels[key]
    )
    train_images[key] = filtered_image
    train_labels[key] = filtered_label

    filtered_image, filtered_label, valid_val[key] = boilerplate._filter_slices(
        val_images[key], val_labels[key]
    )

    val_images[key] = filtered_image
    val_labels[key] = filtered_label

# compute mean and std of the data
all_elements = np.concatenate([train_images[key].flatten() for key in keys])
data_mean = np.mean(all_elements)
data_std = np.std(all_elements)

train_stride = 64
val_stride = 40

# normalizing the data
for key in tqdm(keys, "Normalizing data"):
    train_images[key] = (train_images[key] - data_mean) / data_std
    val_images[key] = (val_images[key] - data_mean) / data_std
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
    model.labeled_ratio = ratio

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
        labeled_ratio=ratio,
        stochastic_block_type=stochastic_block_type,
        n_components=n_components,
        scale=scale,
        use_equivariant=True,
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
