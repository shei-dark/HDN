import sys
sys.path.append('../../../')
sys.path.append('/home/sheida.rahnamai/GIT/HDN/')
import torch
import numpy as np
from sklearn.cluster import HDBSCAN
import random
from tqdm import tqdm
# from lib.dataloader import CustomTestDataset
from lib.dataloader import CustomDataset
import tifffile as tiff
import os
from boilerplate import boilerplate
from scipy.spatial.distance import cdist
from torch.nn.functional import unfold
from sklearn.cluster import KMeans


def get_closest(train, test, metric):

        distances = cdist(test, train, metric=metric)
        closest_index = np.argmin(distances, axis=1)
        return closest_index


def get_all_patches(image, patch_size, stride):
    """
    Efficiently get all patches from the image using unfold.
    """
    image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)  # Ensure float32
    patches = unfold(image_tensor, kernel_size=patch_size, stride=stride)
    patches = patches.permute(0, 2, 1).reshape(-1, patch_size, patch_size)
    return patches

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

dist_metric = ['euclidean', 'cosine', 'correlation', 'seuclidean']
# Euclidean Distance ('euclidean')
# The standard Euclidean distance between two points in Euclidean space.
# Cosine Distance ('cosine')
# Measures the cosine of the angle between two non-zero vectors, commonly used in text analysis.
# Correlation Distance ('correlation')
# Computes the distance based on the Pearson correlation coefficient.
# Seuclidean Distance ('seuclidean')
# The standardized Euclidean distance, taking into account the variance of each feature.

num_clusters = 4
patch_size = 64
sample_size = 500
centre_size = 4
n_channel = 32
hierarchy_level = 3
pad_size = (patch_size - centre_size) // 2
model_dir = "/group/jug/Sheida/HVAE/v01/model/"
model = torch.load(model_dir+"HVAE_best_vae.net")
# train data

data_dir = "/group/jug/Sheida/pancreatic beta cells/download/"
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

One_test_image = ['high_c4']

# Load test image
test_img_path = os.path.join(data_dir, One_test_image[0], f"{One_test_image[0]}_source.tif")
test_images = tiff.imread(test_img_path)

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

train_batch = torch.cat([bg_samples, nucleus_samples, granule_samples, mito_samples], dim=0)
model.mode_pred = True
model.eval()
device = model.device
with torch.no_grad():
   train_batch = train_batch.to(device=device, dtype=torch.float)
   train_batch = train_batch.reshape(-1, 1, patch_size, patch_size)
   output = model(train_batch, model_layers=[i for i in range(hierarchy_level)])
   mu_train = torch.cat([output["mu"][i].view(sample_size * 4, -1) for i in range(hierarchy_level)], dim=1)
   mu_train_mean = torch.cat([torch.mean(mu_train[i*sample_size:(i+1)*sample_size], dim=0).unsqueeze(0) 
                           for i in range(4)], dim=0)
   mu_train_mean = np.array(mu_train_mean.cpu().numpy())

#test data

test_images = (test_images-data_mean)/data_std

for test_index in tqdm(range(1, len(test_images)), desc='Test data'):

   test_slice = test_images[test_index]
   test_slice_lbl = test_ground_truth_image[test_index]
   all_patches = get_all_patches(test_slice, patch_size, 1).unsqueeze(1)
   test_batch_size = 400
   h, w = test_slice.shape
   new_h, new_w = h - patch_size + 1, w - patch_size + 1
   feature_maps = np.zeros((len(dist_metric),(new_h * new_w)))

   model.mode_pred = True
   model.eval()
   device = model.device
   index = 0 
   all_mus = np.zeros(((new_h * new_w), 43008))
   with torch.no_grad():
      all_patches = all_patches.to(device=device, dtype=torch.float)
      for start in tqdm(range(0, all_patches.shape[0], test_batch_size)):
         end = min(start + test_batch_size, all_patches.shape[0])
         output = model(all_patches[start:end], model_layers=[i for i in range(hierarchy_level)])
         mu_test = torch.cat([output["mu"][i].view(end - start, -1) for i in range(hierarchy_level)], dim=1)
         mu_test = np.array(mu_test.cpu().numpy())
         all_mus[index:index + test_batch_size] = mu_test
         for m in range(len(dist_metric)):
            feature_maps[m][index:index + test_batch_size] = get_closest(mu_train_mean, mu_test, metric=dist_metric[m])
         index += test_batch_size

   feature_maps = feature_maps.reshape(len(dist_metric), new_h, new_w)
   # ['euclidean', 'cosine', 'correlation', 'seuclidean']
   for d in range(len(dist_metric)):
      tiff.imwrite(f"{data_dir}/seg/{dist_metric[d]}/seg{test_index}.tif", feature_maps[d])

   while(num_clusters <= 10):
      # Perform K-means clustering
      kmeans = KMeans(n_clusters=num_clusters, random_state=42)
      cluster_labels = kmeans.fit_predict(all_mus)
      clusters = cluster_labels.reshape(new_h, new_w)
      tiff.imwrite(f"{data_dir}/seg/clusters{num_clusters}/seg{test_index}.tif", clusters)
      num_clusters += 1
      