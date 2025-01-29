import os
import warnings

warnings.filterwarnings("ignore")
# We import all our dependencies.
import numpy as np
import torch
from torch.utils.data import DataLoader
from models.lvae import LadderVAE
from boilerplate.dataloader import Custom2DDataset, DynamicSampler
import training
from tqdm import tqdm
import tifffile as tiff

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

patch_size = 64

gaussian_noise_std = None

model_name = "refactoring"
directory_path = "/group/jug/Sheida/HVAE/refactoring/test_00/"

# Model-specific
load_checkpoint = False
checkpoint = "/group/jug/Sheida/HVAE/*_best_vae.net"

noiseModel = None

# Training-specific
batch_size = 512
lr = 3e-5
max_epochs = 100
num_latents = 3
z_dims = [32] * int(num_latents)
blocks_per_layer = 5
batchnorm = True
free_bits = 0.0

alpha = 1  # weight of the inpainting loss
beta = 1e-4  # weight of the KL loss
gamma = 1e-1  # weight of the contrastive loss

initial_mask_size = 1
final_mask_size = 1
initial_label_size = 1
final_label_size = 1
step_interval = 5  # Change every 5 steps


contrastive_learning = True
margin = 50  # distance for negative pairs in contrastive learning
lambda_contrastive = 0.5  # weight of the positive pairs in contrastive learning (1-lambda_contrastive is the weight of the negative pairs)

use_wandb = True

mode = "supervised"
ratio = 1

stochastic_block_type = "normal"  # 'normal' or 'mixture'
conditional = True  # True for conditional LVAE (conditioned on gt label)
condition_type = "mlp"  # 'mlp' or 'transformer'
n_components = 1  # number of components / classes

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
    valid_indices = ~np.all(train_labels[key] == -1, axis=(1, 2))
    train_images[key] = train_images[key][valid_indices]
    train_labels[key] = train_labels[key][valid_indices]
    valid_train[key] = valid_indices

    valid_indices = ~np.all(val_labels[key] == -1, axis=(1, 2))
    val_images[key] = val_images[key][valid_indices]
    val_labels[key] = val_labels[key][valid_indices]
    valid_val[key] = valid_indices

# compute mean and std of the data
all_elements = np.concatenate([train_images[key].flatten() for key in keys])
data_mean = np.mean(all_elements)
data_std = np.std(all_elements)

train_stride = 192  # should be a multiple of 32
val_stride = 120  # should be a multiple of 20

# normalizing the data
for key in tqdm(keys, "Normalizing data"):
    train_images[key] = (train_images[key] - data_mean) / data_std
    val_images[key] = (val_images[key] - data_mean) / data_std

train_set = Custom2DDataset(
    train_images,
    train_labels,
    patch_size,
    initial_label_size,
    train_stride,
    mode,
)
val_set = Custom2DDataset(
    val_images,
    val_labels,
    patch_size,
    initial_label_size,
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
        mask_size=initial_mask_size,
        contrastive_learning=contrastive_learning,
        margin=margin,
        lambda_contrastive=lambda_contrastive,
        labeled_ratio=ratio,
        stochastic_block_type=stochastic_block_type,
        conditional=conditional,
        condition_type=condition_type,
        n_components=n_components,
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
