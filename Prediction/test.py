import os, sys
HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..'))  # parent of myscript/
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import warnings
warnings.filterwarnings("ignore")
# We import all our dependencies.
import numpy as np
import torch
from torch.utils.data import DataLoader
from models.lvae import LadderVAE
from boilerplate.dataloader import BCSSDataset, ModeAwareBalancedAnchorBatchSampler, flex_collate
import training
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from glob import glob
import random
import torch
import matplotlib.pyplot as plt
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

data_dir = "/home/sheida.rahnamai/BCSS/"
img_paths = sorted(glob(data_dir+'images_vahadane/*.png'))
lbl_paths = sorted(glob(data_dir+'masks/*.png'))
imgs = list(np.array(Image.open(path)) for k, path in enumerate(img_paths))
lbls = list(np.array(Image.open(path)) for k, path in enumerate(lbl_paths))

use_wandb = True

patch_size = 64

gaussian_noise_std = None

model_name = "segmentation"
directory_path = "/group/jug/Sheida/HVAE/segmentation/test_BSCC/"

noiseModel = None

# Training-specific
batch_size = 256
lr = 3e-5
max_epochs = 1000
num_latents = 3
z_dims = [32] * int(num_latents)
blocks_per_layer = 5
batchnorm = True
free_bits = 0.0

alpha = 1  # weight of the inpainting loss
beta = 1e-2  # weight of the KL loss
gamma = 1  # weight of the contrastive loss

initial_mask_size = 1
final_mask_size = 1
initial_label_size =1
final_label_size = 1
step_interval = 100

contrastive_learning = True
margin = 50  # distance for negative pairs in contrastive learning
lambda_contrastive = 0.5  # weight of the positive pairs in contrastive learning
# (1-lambda_contrastive is the weight of the negative pairs)

mode = 'supervised' #or 'semisupervised' or 'unsupervised'
labeled_ratio = 1  # ratio of labeled data in semisupervised mode
stochastic_block_type = 'mixture'  # 'normal' or 'mixture'
conditional = True  # True for conditional LVAE (conditioned on gt label)
condition_type = 'mlp'  # 'mlp' or 'transformer'
assert (conditional == True and condition_type != None) or conditional == False
n_components = 18  # number of components for prior
n_classes = 18  # number of classes in the dataset
# train data
keys = list(range(105))

np.random.seed(42)
np.random.shuffle(keys)


# compute mean and std of the data
all_elements = np.concatenate([imgs[key].flatten() for key in keys])
data_mean = np.mean(all_elements)
data_std = np.std(all_elements.astype(np.float32))

sample_ratio = 20

# normalizing the data
for key in tqdm(keys, "Normalizing data"):
    imgs[key] = (imgs[key] - data_mean) / data_std
    
all_labels = set()
label_count = [0] * 21
for k in keys:
    uniq = np.unique(lbls[k])
    for l in uniq:
        label_count[l] += 1
    all_labels.update(uniq.tolist())

print("All unique labels across train dataset:", sorted(all_labels))
# for i, c in enumerate(label_count):
#     print(i, c)
print("Total classes:", len(all_labels))

train_set = BCSSDataset(
    images=imgs,    
    labels=lbls,
    patch_size=patch_size,
    label_size=initial_label_size,
    mode='supervised',
    ratio=labeled_ratio,
)

len(train_set)

train_loader = DataLoader(
    train_set,
    batch_sampler=ModeAwareBalancedAnchorBatchSampler(
        train_set, total_patches_per_batch=batch_size, shuffle=True
    ),
    collate_fn=flex_collate,
)


img_shape = (3, 64, 64)

model = LadderVAE(
    z_dims=z_dims,
    blocks_per_layer=blocks_per_layer,
    data_mean=data_mean,
    data_std=data_std,
    noiseModel=noiseModel,
    conv_mult=2,
    color_ch=3,
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
    val_loader=train_loader,
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
