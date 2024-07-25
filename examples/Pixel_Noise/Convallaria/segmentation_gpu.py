import sys
sys.path.append('../../../')
sys.path.append('/home/sheida.rahnamai/GIT/HDN/')
import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np
from tifffile import imread
import glob
import matplotlib.patches as patches
import hdbscan as hd
from hdbscan import prediction
from sklearn.cluster import HDBSCAN
from collections import Counter
from openTSNE import TSNE
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
# %reload lib.plotting 
from lib import plotting as p
from lib import dataprep as dp
import random
import umap
from tqdm import tqdm
from lib.evaluation import FeatureExtractor
# from lib.dataloader import CustomTestDataset
from lib.dataloader import CustomDataset
import tifffile as tiff
from sklearn.utils import shuffle
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import os
from boilerplate import boilerplate
from importlib import reload
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

patch_size = 64
sample_size = 2
centre_size = 4
n_channel = 32
hierarchy_level = 3
pad_size = (patch_size - centre_size) // 2
model_dir = "/group/jug/Sheida/HVAE/"
model = torch.load(model_dir+"HVAE_best_vae.net")

# train data

data_dir="/group/jug/Sheida/pancreatic beta cells/download/"
Three_train_images = ['high_c1', 'high_c2', 'high_c3']

# Load source images
train_img_paths = [os.path.join(data_dir, img, f"{img}_source.tif") for img in Three_train_images]
images = {img: tiff.imread(path) for img, path in zip(Three_train_images, train_img_paths)}

# Print loaded train images paths
print("Train images loaded from paths:")
for img, path in zip(Three_train_images, train_img_paths):
   print(path)

labels = {}
# Load ground truth images
for img in Three_train_images:
   gt_path = os.path.join(data_dir, img, f"{img}_gt.tif")
   labels[img] = tiff.imread(gt_path)

# Initialize dictionaries for the split data
train_images = {}
val_images = {}
train_labels = {}
val_labels = {}

keys = ['high_c1', 'high_c2', 'high_c3']

for key in tqdm(keys, desc='Splitting data'):
   filtered_image, filtered_label = boilerplate._filter_slices(images[key], labels[key])
   train_image, val_image, train_label, val_label = boilerplate._split_slices(
      filtered_image, filtered_label
   )
   train_images[key] = train_image
   val_images[key] = val_image
   train_labels[key] = train_label
   val_labels[key] = val_label

# compute mean and std of the data
all_elements = np.concatenate([train_images[key].flatten() for key in keys])
data_mean = np.mean(all_elements)
data_std = np.std(all_elements)

# normalizing the data
for key in tqdm(keys, 'Normalizing data'):
   train_images[key] = (train_images[key] - data_mean) / data_std
   val_images[key] = (val_images[key] - data_mean) / data_std

train_set = CustomDataset(train_images, train_labels)
val_set = CustomDataset(val_images, val_labels)

One_test_image = ['high_c4']

# Load test image
test_img_path = os.path.join(data_dir, One_test_image[0], f"{One_test_image[0]}_source.tif")
test_image = tiff.imread(test_img_path)

# Print loaded test images paths
print("Test image loaded from path:")
print(test_img_path)

# Load test ground truth images
test_gt_path = os.path.join(data_dir, One_test_image[0], f"{One_test_image[0]}_gt.tif")
test_ground_truth_image = tiff.imread(test_gt_path)

bg_indices = random.sample(train_set.patches_by_label[0],sample_size)
nucleus_indices = random.sample(train_set.patches_by_label[1],sample_size)
granule_indices = random.sample(train_set.patches_by_label[2],sample_size)
mito_indices = random.sample(train_set.patches_by_label[3],sample_size)

bg_samples, bg_cls, bg_lbl = train_set[bg_indices]
nucleus_samples, nucleus_cls, nucleus_lbl = train_set[nucleus_indices]
granule_samples, granule_cls, granule_lbl = train_set[granule_indices]
mito_samples, mito_cls, mito_lbl = train_set[mito_indices]

bg_samples = bg_samples.squeeze(1)
bg_lbl = bg_lbl.squeeze(1)
nucleus_samples = nucleus_samples.squeeze(1)
nucleus_lbl = nucleus_lbl.squeeze(1)
granule_samples = granule_samples.squeeze(1)
granule_lbl = granule_lbl.squeeze(1)
mito_samples = mito_samples.squeeze(1)
mito_lbl = mito_lbl.squeeze(1)

#test data

test_images = (test_image-data_mean)/data_std

test_slice = torch.tensor(test_images[626])
test_slice_lbl = test_ground_truth_image[626]

train_batch = torch.cat([bg_samples, nucleus_samples, granule_samples, mito_samples], dim=0)
FE = FeatureExtractor(model, patch_size, centre_size)
model.mode_pred = True
model.eval()
device = model.device
feature_map = []

with torch.no_grad():
    train_batch = train_batch.to(device=device, dtype=torch.float)
    train_batch = train_batch.reshape(-1, 1, patch_size, patch_size)
    output = model(train_batch, train_batch, train_batch, model_layers=[i for i in range(hierarchy_level)])
    mus = output["mu"]
    for i in range(len(mus[0])):
        temp = mus[0][i].flatten()
        for j in range(1, len(mus)):
            temp = torch.cat((temp, mus[j][i].flatten()), dim=-1)
        feature_map.append(temp.cpu().numpy())

    train_mu = np.vstack(feature_map)

H, W = test_slice.shape
pred = torch.zeros((H, W), device=device)
latent_embedding = torch.zeros((H, W, 43008), dtype=torch.float32)

for i in tqdm(range(0, H - patch_size, 1)):
    for j in range(0, W - patch_size, 1):
        pred[i+30:i+34, j+30:j+34] += 1
        latent_embedding[i+30:i+34, j+30:j+34, :] += (FE.get_feature_maps(test_slice[i:i+patch_size, j:j+patch_size]) / 16)

pred = pred.cpu().numpy()
coords = np.column_stack(np.where(pred == 16))
min_row, min_col = coords.min(axis=0)
max_row, max_col = coords.max(axis=0)
label_pred = np.zeros((max_row - min_row + 1, max_col - min_col + 1))

for i in tqdm(range(min_row, max_row + 1)):
    for j in range(min_col, max_col + 1):
        label_pred[i - min_row, j - min_col] = FE.get_closest(train_mu, latent_embedding[i, j].cpu().numpy())

cropped_image = pred[min_row:max_row + 1, min_col:max_col + 1]
print(min_row, max_row, min_col, max_col)
dp.save_pickle('label_pred1.pkl', label_pred)
