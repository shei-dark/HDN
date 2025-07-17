import sys
import time
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
# img_idx = range(49,1016)
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
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=1
    )
    for model_v in model_versions:
        model = torch.load(model_dir + model_v + "/model_supervised/segmentation_best_vae.net")
        data_mean = model.data_mean
        data_std = model.data_std
        model.mode_pred = True
        model.eval()
        device = model.device
        print(f"Processing image slice {test_index} with model version {model_v}")
        index = 0
        
        pred = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                batch = batch.to(device)
                batch = (batch - data_mean) / data_std
                output = model(batch)
                y_pred = output["pi"].argmax(dim=-1) 
                pred.extend(y_pred.cpu().numpy())

        pred_array = np.array(pred)

        clusters = pred_array.reshape(
            test_dataset.num_patches_y, test_dataset.num_patches_x
        )
        tiff.imwrite(f"{model_dir}{model_v}/seg/{test_index}_sup.tif", clusters.astype(np.uint8))
        print(f"Segmentation for image slice {test_index} saved")