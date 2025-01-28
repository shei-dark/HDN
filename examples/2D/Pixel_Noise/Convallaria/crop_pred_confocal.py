import sys
import time

sys.path.append("../../../")
sys.path.append("/home/sheida.rahnamai/GIT/HDN/")
import torch
import numpy as np
from tqdm import tqdm

# from lib.dataloader import CustomTestDataset
from boilerplate.dataloader import CustomTestDataset
import tifffile as tiff
import os
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

dist_metric = ["cosine"]

num_clusters = 4
patch_size = (3, 64, 64)
hierarchy_level = 3
data_dir = "/group/jug/Sheida/Microsim/macrophage/"
key = "simulated_jrc_macrophage-2"

img = tiff.imread(data_dir + key + "_source.tif")
lbl = tiff.imread(data_dir + key + "_gt.tif")

num_stacks = img.shape[1]
indices = np.arange(num_stacks)
np.random.seed(42)  # For reproducibility
np.random.shuffle(indices)
# Calculate split sizes
train_size = int(0.7 * num_stacks)  # 70%
val_size = int(0.15 * num_stacks)  # 15%
test_size = num_stacks - train_size - val_size  # Remaining
# Split indices

test_indices = indices[train_size + val_size :]

# Split the stack along the second dimension (stacks)
test_images = img[:, test_indices, :, :]
test_ground_truth_image = lbl[:, test_indices, :, :]

# compute mean and std of the data

model_dir = "/group/jug/Sheida/HVAE/gmvae/"
img_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8]
model_versions = ["confocal_mlp_semi"]
batch_size = 1024


for test_index in tqdm(img_idx):
    print("Processing test dataset")
    test_dataset = CustomTestDataset(
        test_images, patch_size=(3, 64, 64), index=test_index, stride=1, model="4D"
    )
    print("Test dataset loaded. Processing test dataloader")
    dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    for model_v in model_versions:
        model = torch.load(model_dir + model_v + "/model/confocal_best_vae.net")
        data_mean = model.data_mean
        data_std = model.data_std
        model.mode_pred = True
        model.eval()
        device = model.device
        print(f"Processing image slice {test_index} with model version {model_v}")
        index = 0

        # all_mus = np.zeros(
        #     ((test_dataset.num_patches_y * test_dataset.num_patches_x), 49152),
        #     dtype=np.float16,
        # )
        all_mus = np.zeros(
            ((test_dataset.num_patches_y * test_dataset.num_patches_x), 43008),
            dtype=np.float16,
        )
        pred = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                batch = batch.to(device)
                batch = (batch - data_mean) / data_std
                output = model(batch)
                # mu_test = torch.cat([output["mu"][i].reshape(batch.shape[0], -1) for i in range(hierarchy_level)], dim=1)
                y_pred = output["pi"].argmax(dim=-1)
                # mu_test = np.array(mu_test.cpu().numpy())
                # all_mus[index : index + batch.shape[0]] = mu_test
                # index += batch.shape[0]
                pred.extend(y_pred.cpu().numpy())

        # Perform K-means clustering
        # kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        # print("Fitting K-means")
        # start_time = time.time()
        # cluster_labels = kmeans.fit_predict(all_mus)
        # end_time = time.time()
        # print("K-means fitted in {:.2f} seconds".format(end_time - start_time))
        # clusters = cluster_labels.reshape(
        #     test_dataset.num_patches_y, test_dataset.num_patches_x
        # )
        pred_array = np.array(pred)

        clusters = pred_array.reshape(
            test_dataset.num_patches_y, test_dataset.num_patches_x
        )
        tiff.imwrite(f"{model_dir}{model_v}/seg/{test_index}.tif", clusters)
        print(f"Segmentation for image slice {test_index} saved")
