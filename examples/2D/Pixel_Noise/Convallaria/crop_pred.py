import sys
import time
sys.path.append("../../../")
sys.path.append("/home/sheida.rahnamai/GIT/My_Plugin/epsSeg/")
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


num_clusters = 4
patch_size = (64, 64)
hierarchy_level = 3
data_dir = "/group/jug/Sheida/pancreatic beta cells/download/"

One_test_image = ["high_c4"]

# Load test image
test_img_path = os.path.join(
    data_dir, One_test_image[0], f"{One_test_image[0]}_source.tif"
)
test_images = tiff.imread(test_img_path)
# Print loaded test images paths
print("Test image loaded from path:")
print(test_img_path)

# Load test ground truth images
test_gt_path = os.path.join(data_dir, One_test_image[0], f"{One_test_image[0]}_gt.tif")
test_ground_truth_image = tiff.imread(test_gt_path)
model_dir = "/group/jug/Sheida/HVAE/segmentation/"
img_idx = [626]
model_versions = ["06"]
batch_size = 512


for test_index in tqdm(img_idx):
    print("Processing test dataset")
    test_dataset = CustomTestDataset(
        test_images, patch_size=patch_size, index=test_index, stride=1, model="2D"
    )
    print("Test dataset loaded. Processing test dataloader")
    dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    for model_v in model_versions:
        model = torch.load(model_dir + model_v + "/model_unsupervised/segmentation_best_vae.net")
        data_mean = model.data_mean
        data_std = model.data_std
        model.mode_pred = True
        model.eval()
        device = model.device
        print(f"Processing image slice {test_index} with model version {model_v}")
        index = 0
        
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
                mu_list = []
                for mu in output['mu']:  # Iterate over all 3 levels
                    batch_size, channels, height, width = mu.shape
                    
                    # Fully flatten (merge spatial and channel dimensions)
                    mu_flattened = mu.view(batch_size, -1)  # Shape: (batch_size, channels * height * width)
                    mu_list.append(mu_flattened)

                # Concatenate across all levels
                mu_concat = torch.cat(mu_list, dim=1)  # (batch_size, total_features)
                all_mus[index:index+batch.shape[0]] = mu_concat.cpu().numpy()
                index+=batch.shape[0]

        # Stack all batches
        # all_mus = np.concatenate(all_mus, axis=0)  # Shape: (total_patches, total_features)

        num_patches_y, num_patches_x = test_dataset.num_patches_y, test_dataset.num_patches_x

        # Perform clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(all_mus)

        # Reshape cluster labels to match the image shape
        clustered_image = cluster_labels.reshape((num_patches_y, num_patches_x))

        
        tiff.imwrite(f"{model_dir}{model_v}/seg/{test_index}_kmeans.tif", clustered_image.astype(np.uint8))
        print(f"Segmentation for image slice {test_index} saved")
        