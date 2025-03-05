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
from boilerplate.dataloader import Custom2DDataset, DynamicSampler
import training
from tqdm import tqdm
import tifffile as tiff

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--directory_path", type=str, default="/group/jug/Sheida/HVAE/segmentation/test/"
)
parser.add_argument("--contrastive_learning", type=bool, default=True)
parser.add_argument("--mode", type=str, default="supervised")
parser.add_argument("--labeled_ratio", type=float, default=0.75)
parser.add_argument("--stochastic_block_type", type=str, default="mixture")
parser.add_argument("--conditional", type=bool, default=True)
parser.add_argument("--condition_type", type=str, default="mlp")
parser.add_argument("--sample_ratio", type=int, default=5)
parser.add_argument("--num_latents", type=int, default=3)
parser.add_argument("--blocks_per_layer", type=int, default=5)
parser.add_argument("--alpha", type=float, default=1)
parser.add_argument("--beta", type=float, default=1e-2)
parser.add_argument("--gamma", type=float, default=1e-2)
parser.add_argument("--initial_mask_size", type=int, default=3)
parser.add_argument("--final_mask_size", type=int, default=3)
parser.add_argument("--initial_label_size", type=int, default=3)
parser.add_argument("--final_label_size", type=int, default=3)
parser.add_argument("--step_interval", type=int, default=10)
parser.add_argument("--load_checkpoint", type=bool, default=False)

args = parser.parse_args()
use_wandb = True

patch_size = 64

gaussian_noise_std = None

model_name = "segmentation"
directory_path = args.directory_path

# Model-specific
load_checkpoint = args.load_checkpoint
checkpoint = directory_path + "model_supervised/segmentation_best_vae.net"

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
labeled_ratio = args.labeled_ratio  # ratio of labeled data in semisupervised mode
stochastic_block_type = args.stochastic_block_type  # 'normal' or 'mixture'
conditional = args.conditional  # True for conditional LVAE (conditioned on gt label)
condition_type = args.condition_type  # 'mlp' or 'transformer'
assert (conditional == True and condition_type != None) or conditional == False
n_components = 4  # number of components for prior
n_classes = 4  # number of classes in the dataset
# train data
data_dir = "/group/jug/Sheida/pancreatic beta cells/download/"
keys = ["high_c1", "high_c2", "high_c3"]

img_paths = [os.path.join(data_dir + key + f"/{key}_source.tif") for key in keys]
lbl_paths = [os.path.join(data_dir + key + f"/{key}_gt.tif") for key in keys]
imgs = {key: tiff.imread(path) for key, path in zip(keys, img_paths)}
lbls = {key: tiff.imread(path) for key, path in zip(keys, lbl_paths)}
train_images, val_images, train_labels, val_labels = {}, {}, {}, {}

np.random.seed(42)
for key in keys:
    total_samples = imgs[key].shape[0]

    # Create shuffled indices
    indices = np.arange(total_samples)
    np.random.shuffle(indices)  # Shuffles in place

    # Compute split index
    split_idx = int(0.8 * total_samples)

    # Split the indices
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]

    # Use shuffled indices to assign train/val splits
    train_images[key] = imgs[key][train_idx]
    val_images[key] = imgs[key][val_idx]
    train_labels[key] = lbls[key][train_idx]
    val_labels[key] = lbls[key][val_idx]

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

sample_ratio = args.sample_ratio

# normalizing the data
for key in tqdm(keys, "Normalizing data"):
    train_images[key] = (train_images[key] - data_mean) / data_std
    val_images[key] = (val_images[key] - data_mean) / data_std

train_set = Custom2DDataset(
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

val_set = Custom2DDataset(
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
print(f"Train set: {len(train_set)}, Val set: {len(val_set)}")
print(
    f"unrecognized: {len(train_set.patches_by_label[0])}, unrecognized: {len(val_set.patches_by_label[0])}"
)
print(
    f"nucleus: {len(train_set.patches_by_label[1])}, nucleus: {len(val_set.patches_by_label[1])}"
)
print(
    f"granule: {len(train_set.patches_by_label[2])}, granule: {len(val_set.patches_by_label[2])}"
)
print(
    f"mitochondria: {len(train_set.patches_by_label[3])}, mitochondria: {len(val_set.patches_by_label[3])}"
)

train_sampler = DynamicSampler(train_set, batch_size, labeled_ratio=labeled_ratio)
val_sampler = DynamicSampler(val_set, batch_size, labeled_ratio=labeled_ratio)

train_loader = DataLoader(
    train_set, sampler=train_sampler, num_workers=8, prefetch_factor=4, pin_memory=True
)
val_loader = DataLoader(
    val_set, sampler=val_sampler, num_workers=8, prefetch_factor=4, pin_memory=True
)

img_shape = (64, 64)

if load_checkpoint:
    model = torch.load(checkpoint)
    model.update_mode("semisupervised")

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
