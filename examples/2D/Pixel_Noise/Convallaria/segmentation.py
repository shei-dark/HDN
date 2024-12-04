import sys
import time
sys.path.append("../../../")
sys.path.append("/home/sheida.rahnamai/GIT/HDN/")
import torch
import numpy as np
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans

# from lib.dataloader import CustomTestDataset
from boilerplate.dataloader import CustomTestDataset
import tifffile as tiff
import os
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

dist_metric = ["euclidean"]

num_clusters = 4
patch_size = (1, 64, 64)
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
model_dir = "/group/jug/Sheida/HVAE/TAC/"
img_idx = np.arange(48, 1015)
model_versions = ["ex_15"]
batch_size = 1024

# kmeans = MiniBatchKMeans(
#     n_clusters=num_clusters, random_state=42, batch_size=batch_size
# )

for test_index in tqdm(img_idx):
    print("Processing test dataset")
    test_dataset = CustomTestDataset(
        test_images, patch_size=(64, 64), index=test_index, stride=1, model="2D"
    )
    # test_dataset = CustomTestDataset(
    #     test_images, patch_size=patch_size, index=test_index, model="2D"
    # )
    print("Test dataset loaded. Processing test dataloader")
    dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    for model_v in model_versions:
        model = torch.load(model_dir + model_v + "/model/EXTAC_best_vae.net")
        data_mean = model.data_mean
        data_std = model.data_std
        model.mode_pred = True
        model.eval()
        device = model.device
        print(f"Processing image slice {test_index} with model version {model_v}")
        index = 0
        all_mus = np.zeros(
            ((test_dataset.num_patches_y * test_dataset.num_patches_x), 49152),
            dtype=np.float16,
        )
        with torch.no_grad():
            for batch in tqdm(dataloader):
                batch = batch.to(device)
                batch = (batch - data_mean) / data_std
                output = model(batch)
                # mu_test = torch.cat(
                #     [
                #         output["mu"][i].reshape(batch.shape[0], -1)
                #         for i in range(hierarchy_level)
                #     ],
                #     dim=1,
                # )
                # mu_test = output["mu"][-1].reshape(batch.shape[0], -1)
                mu_test = torch.cat([output["mu"][i].reshape(batch.shape[0], -1) for i in range(hierarchy_level)], dim=1)
                mu_test = np.array(mu_test.cpu().numpy())
                all_mus[index : index + batch.shape[0]] = mu_test
                index += batch.shape[0]

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        print("Fitting K-means")
        start_time = time.time()
        cluster_labels = kmeans.fit_predict(all_mus)
        end_time = time.time()
        print("K-means fitted in {:.2f} seconds".format(end_time - start_time))
        clusters = cluster_labels.reshape(
            test_dataset.num_patches_y, test_dataset.num_patches_x
        )
        tiff.imwrite(f"{model_dir}{model_v}/seg/{test_index}.tif", clusters)
        print(f"Segmentation for image slice {test_index} saved")
