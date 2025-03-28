import os
import argparse

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import warnings

warnings.filterwarnings("ignore")
# We import all our dependencies.
import numpy as np
import torch
from torch.utils.data import DataLoader
from models.lvae import LadderVAE
from boilerplate.dataloader import CustomLightDataset, DynamicSampler
import training
from tqdm import tqdm
import tifffile as tiff
from glob import glob
from aicsimageio import AICSImage, imread_dask


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--directory_path", type=str, default="/group/jug/Sheida/HVAE/experiments/test/"
)
parser.add_argument("--contrastive_learning", type=bool, default=True)
parser.add_argument("--mode", type=str, default="supervised")
parser.add_argument("--labeled_ratio", type=float, default=0.75)
parser.add_argument("--stochastic_block_type", type=str, default="mixture")
parser.add_argument("--conditional", type=bool, default=True)
parser.add_argument("--condition_type", type=str, default="mlp")
parser.add_argument("--sample_ratio", type=int, default=1000)
parser.add_argument("--num_latents", type=int, default=2)
parser.add_argument("--blocks_per_layer", type=int, default=3)
parser.add_argument("--alpha", type=float, default=1)
parser.add_argument("--beta", type=float, default=1e-2)
parser.add_argument("--gamma", type=float, default=1e-2)
parser.add_argument("--initial_mask_size", type=int, default=3)
parser.add_argument("--final_mask_size", type=int, default=3)
parser.add_argument("--initial_label_size", type=int, default=3)
parser.add_argument("--final_label_size", type=int, default=3)
parser.add_argument("--step_interval", type=int, default=10)
parser.add_argument("--load_checkpoint", type=bool, default=False)
parser.add_argument("--checkpoint", type=str, default="")


args = parser.parse_args()
use_wandb = True

patch_size = 64

gaussian_noise_std = None

model_name = "experiments"
directory_path = args.directory_path

# Model-specific
load_checkpoint = args.load_checkpoint
checkpoint = args.checkpoint

noiseModel = None

# Training-specific
batch_size = 1024
lr = 3e-5
max_epochs = 300
num_latents = args.num_latents
z_dims = [32] * int(num_latents)
blocks_per_layer = args.blocks_per_layer
batchnorm = True
free_bits = 0.0

alpha = args.alpha  # weight of the inpainting loss
beta = args.beta  # weight of the KL loss
gamma = args.gamma  # weight of the contrastive loss

initial_mask_size = args.initial_mask_size
final_mask_size = args.final_mask_size
initial_label_size = args.initial_label_size
final_label_size = args.final_label_size
step_interval = args.step_interval

contrastive_learning = args.contrastive_learning
margin = 50  # distance for negative pairs in contrastive learning
lambda_contrastive = 0.5  # weight of the positive pairs in contrastive learning
# (1-lambda_contrastive is the weight of the negative pairs)

mode = args.mode  # 'supervised' or 'semisupervised' or 'unsupervised'
labeled_ratio = args.labeled_ratio
stochastic_block_type = args.stochastic_block_type  # 'normal' or 'mixture'
conditional = args.conditional  # True for conditional LVAE (conditioned on gt label)
condition_type = args.condition_type  # 'mlp' or 'transformer'
assert (conditional == True and condition_type != None) or conditional == False
n_components = 3  # number of components for prior
n_classes = 3  # number of classes in the dataset
# train data

data_dir = '/group/jug/Enrico/TISSUE_roi/'
train_dirs = sorted(glob(data_dir + "training/*"))

images = []  # Will hold arrays of shape (2, H, W), varying sizes
labels = []  # Will hold arrays of shape (H, W), varying sizes

for img_dir in train_dirs:
    cell_name = img_dir.split('/')[-1]

    channel_0_path = f"{img_dir}/{cell_name} - C=0.tif"
    channel_1_path = f"{img_dir}/{cell_name} - C=1.tif"
    mask_path = f"{img_dir}/{cell_name}_CELLS.tif"

    # Lazy load using dask
    ch0_lazy = imread_dask(channel_0_path)[0, 0, 0]  # assuming STCZYX and you have only one slice
    ch1_lazy = imread_dask(channel_1_path)[0, 0, 0]
    mask_lazy = imread_dask(mask_path)[0, 0, 0]

    # Compute arrays only when necessary
    ch0 = ch0_lazy.compute().astype(np.uint16)
    ch1 = ch1_lazy.compute().astype(np.uint16)
    mask = mask_lazy.compute().astype(np.uint16)

    stacked_channels = np.stack([ch0, ch1], axis=0)

    images.append(stacked_channels)
    labels.append(mask)

print(f"Images loaded (lazy): {len(images)}")
print(f"Shape of first lazy-loaded image: {images[0].shape}")
data_dir = "/group/jug/Enrico/TISSUE/"
train_img_paths = sorted(glob(data_dir + "training/*"))
train_images = tiff.imread(train_img_paths).astype(np.float32)
train_gt_paths = sorted(glob(data_dir + "gt/train/*.tif"))
train_labels = tiff.imread(train_gt_paths)
val_img_paths = sorted(glob(data_dir + "validation/*.tif"))
val_images = tiff.imread(val_img_paths).astype(np.float32)
val_gt_paths = sorted(glob(data_dir + "gt/val/*.tif"))
val_labels = tiff.imread(val_gt_paths)

train_labels[train_labels == 3] = 1
val_labels[val_labels == 3] = 1

# compute mean and std of the data
# all_elements = .flatten()
data_mean_cell = np.mean(train_images[:,0,:,:])
data_std_cell = np.std(train_images[:,0,:,:])
data_mean_nuclei = np.mean(train_images[:,1,:,:])
data_std_nuclei = np.std(train_images[:,1,:,:])

sample_ratio = args.sample_ratio

# normalizing the data
train_images[:,0,:,:] = (train_images[:,0,:,:] - data_mean_cell) / data_std_cell
train_images[:,1,:,:] = (train_images[:,1,:,:] - data_mean_nuclei) / data_std_nuclei
val_images[:,0,:,:] = (val_images[:,0,:,:] - data_mean_cell) / data_std_cell
val_images[:,1,:,:] = (val_images[:,1,:,:] - data_mean_nuclei) / data_std_nuclei

train_set = CustomLightDataset(
    images=train_images,
    labels=train_labels,
    patch_size=patch_size,
    label_size=initial_label_size,
    mode=mode,
    n_classes=n_classes,
    sampling_ratio=sample_ratio,
    ignore_lbl=-1,
    ratio=labeled_ratio,
)

val_set = CustomLightDataset(
    images=val_images,
    labels=val_labels,
    patch_size=patch_size,
    label_size=initial_label_size,
    mode=mode,
    n_classes=n_classes,
    sampling_ratio=sample_ratio,
    ignore_lbl=-1,
    ratio=labeled_ratio,
)

print(f'Train set: {len(train_set)}, Val set: {len(val_set)}')
print(f"background: {len(train_set.patches_by_label[0])}, background: {len(val_set.patches_by_label[0])}")
print(f"cell: {len(train_set.patches_by_label[1])}, cell: {len(val_set.patches_by_label[1])}")
print(f"nuclei: {len(train_set.patches_by_label[2])}, nuclei: {len(val_set.patches_by_label[2])}")

train_sampler = DynamicSampler(train_set, batch_size)
val_sampler = DynamicSampler(val_set, batch_size)

train_loader = DataLoader(train_set, sampler=train_sampler, num_workers=8, prefetch_factor=4, pin_memory=True)
val_loader = DataLoader(val_set, sampler=val_sampler, num_workers=8, prefetch_factor=4, pin_memory=True)

img_shape = (64, 64)

if load_checkpoint:
    model = torch.load(checkpoint)
    model.update_mode("semisupervised")

else:
    model = LadderVAE(
        z_dims=z_dims,
        blocks_per_layer=blocks_per_layer,
        data_mean=data_mean_cell,
        data_std=data_std_cell,
        noiseModel=noiseModel,
        conv_mult=2,
        color_ch=2,
        device=device,
        batchnorm=batchnorm,
        free_bits=free_bits,
        img_shape=img_shape,
        grad_checkpoint=True,
        mask_size=initial_mask_size,
        contrastive_learning=contrastive_learning,
        margin=margin,
        lambda_contrastive=lambda_contrastive,
        stochastic_block_type=stochastic_block_type,
        conditional=conditional,
        condition_type=condition_type,
        n_components=n_components,
        training_mode=mode,
        labeled_ratio=labeled_ratio,
    ).cuda()
print(model)
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
    gradient_scale=256,
    use_wandb=use_wandb,
    max_grad_norm=1,
    initial_label_size=initial_label_size,
    final_label_size=final_label_size,
    initial_mask_size=initial_mask_size,
    final_mask_size=final_mask_size,
    step_interval=step_interval,
)
