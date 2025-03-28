import torch
import torch.onnx
import numpy as np
from boilerplate.dataloader import CustomTestDataset
import tifffile as tiff
import os
from torch.utils.data import DataLoader
import time
import datetime
from torch.amp import autocast
from sklearn.cluster import KMeans

dist_metric = ["euclidean"]
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print("Device: ", device)
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
model_dir = "/group/jug/Sheida/HVAE/segmentation/"
img_idx = list(range(142, 1016))
model_versions = ["06"] #TODO
batch_size = 2048

max_step = len(img_idx) * len(model_versions)
step = 0
seconds_last = time.time()
for model_v in model_versions:
    # onnx_file_path = model_dir + model_v + "segmentation_model.onnx"

    model = torch.load(
        model_dir + model_v + "/model_supervised/segmentation_best_vae.net", #TODO
        weights_only=False,
    )
        
    data_mean = model.data_mean
    data_std = model.data_std
    model.mode_pred = True
    model.eval().to(device)

    device = model.device
    for test_index in img_idx:
        test_dataset = CustomTestDataset(
            test_images, patch_size=(64, 64), index=test_index, stride=1, model="2D"
        )
        dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )
        print(f"Processing image slice {test_index} with model version {model_v}")
        index = 0
        all_mus = np.zeros(
            ((test_dataset.num_patches_y * test_dataset.num_patches_x), 43008),
            dtype=np.float16,
        )
        pred = []
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(device, non_blocking=True)
                batch = batch.float()  
                batch.sub_(data_mean).div_(data_std)  # In-place normalization (faster)
                with autocast(device_type='cuda'):  # Enable mixed precision
                    output = model(batch)

                # y_pred = output["pi"].argmax(dim=-1)
                # pred.extend(y_pred.cpu().numpy())
                mu_test = torch.cat([output["mu"][i].reshape(batch.shape[0], -1) for i in range(hierarchy_level)], dim=1)
                mu_test = np.array(mu_test.cpu().numpy())
                all_mus[index : index + batch.shape[0]] = mu_test
                index += batch.shape[0]

        # pred_array = np.array(pred)

        # clusters = pred_array.reshape(
            # test_dataset.num_patches_y, test_dataset.num_patches_x
        # )
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)

        print("Fitting K-means")
        start_time = time.time()
        cluster_labels = kmeans.fit_predict(all_mus)
        end_time = time.time()
        print("K-means fitted in {:.2f} seconds".format(end_time - start_time))
        clusters = cluster_labels.reshape(
            test_dataset.num_patches_y, test_dataset.num_patches_x
        )
        tiff.imwrite(f"{model_dir}{model_v}/seg_clustering/{test_index}.tif", clusters.astype(np.uint8))
        print(f"Segmentation for image slice {test_index} saved")

        # seg_dir = f"{model_dir}{model_v}/seg_supervised_beta/" #TODO
        # os.makedirs(seg_dir, exist_ok=True)
        # tiff.imwrite(f"{seg_dir}{test_index}.tif", clusters.astype(np.uint8))
        # print(
            # f"Segmentation for image slice {test_index} with model {model_v} is saved"
        # )

        # seconds = time.time()
        # secondsElapsed = float(seconds - seconds_last)
        # seconds_last = seconds
        # remainingEps = (max_step + 1) - (step + 1)
        # estRemainSecondsInt = int(secondsElapsed) * (remainingEps)
        # print("Time for epoch: " + str(int(secondsElapsed)) + "seconds")

        # print(
        #     "Est remaining time: "
        #     + str(datetime.timedelta(seconds=estRemainSecondsInt))
        #     + " or "
        #     + str(estRemainSecondsInt)
        #     + " seconds"
        # )

        print("----------------------------------------", flush=True)
        step += 1
