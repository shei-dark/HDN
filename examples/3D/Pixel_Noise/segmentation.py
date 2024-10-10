import sys

sys.path.append("../../../")
sys.path.append("/home/sheida.rahnamai/GIT/HDN/")
import torch
import numpy as np
from sklearn.cluster import HDBSCAN
from tqdm import tqdm

# from lib.dataloader import CustomTestDataset
from boilerplate.dataloader import Custom3DTestDataset
import tifffile as tiff
import os
from sklearn.cluster import KMeans

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

dist_metric = ["cosine"]

num_clusters = 4
patch_size = (8, 64, 64)
centre_size = 5
n_channel = 32
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

model_dir = "/group/jug/Sheida/HVAE/3D/"
img_idx = [500,600,700]
model_versions = [4,5,6,7,8]
batch_size = 256
for test_index in img_idx:
    print("Processing test dataset")
    test_dataset = Custom3DTestDataset(
        test_images, patch_size=patch_size, index=test_index
    )
    print("Test dataset loaded. Processing test dataloader")
    dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    for model_v in model_versions:
        model = torch.load(
            model_dir + "v0" + str(model_v) + "/model/3D_HVAE_best_vae.net"
        )
        data_mean = model.data_mean
        data_std = model.data_std
        model.mode_pred = True
        model.eval()
        device = model.device
        print(f"Processing image slice {test_index} with model version {model_v}")
        index = 0
        all_mus = np.zeros(
            ((test_dataset.height * test_dataset.width), 43008), dtype=np.float16
        )
        with torch.no_grad():
            for batch in tqdm(dataloader):
                batch = batch.to(device)
                batch = (batch - data_mean) / data_std
                output = model(batch)
                mu_test = torch.cat(
                    [
                        torch.mean(output["mu"][i], dim=2).reshape(batch.shape[0], -1)
                        for i in range(hierarchy_level)
                    ],
                    dim=1,
                )
                mu_test = np.array(mu_test.cpu().numpy())
                all_mus[index : index + batch.shape[0]] = mu_test
                index += batch.shape[0]

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(all_mus)
        clusters = cluster_labels.reshape(test_dataset.height, test_dataset.width)
        tiff.imwrite(f"{model_dir}v0{model_v}/{test_index}.tif", clusters)
