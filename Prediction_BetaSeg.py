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
img_idx = list(range(49, 1016))
model_versions = ["04"]
batch_size = 1024

max_step = len(img_idx) * len(model_versions)
step = 0
seconds_last = time.time()
for model_v in model_versions:
    onnx_file_path = model_dir + model_v + "segmentation_model.onnx"

    model = torch.load(
        model_dir + model_v + "/model_supervised/segmentation_best_vae.net",
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

        pred = []
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(device, non_blocking=True)
                batch = batch.float()  
                batch.sub_(data_mean).div_(data_std)  # In-place normalization (faster)
                with autocast(device_type='cuda'):  # Enable mixed precision
                    output = model(batch)

                y_pred = output["pi"].argmax(dim=-1)
                pred.extend(y_pred.cpu().numpy())

        pred_array = np.array(pred)

        clusters = pred_array.reshape(
            test_dataset.num_patches_y, test_dataset.num_patches_x
        )
        seg_dir = f"{model_dir}{model_v}/seg_supervised/"
        os.makedirs(seg_dir, exist_ok=True)
        tiff.imwrite(f"{seg_dir}{test_index}.tif", clusters.astype(np.uint8))
        print(
            f"Segmentation for image slice {test_index} with model {model_v} is saved"
        )

        seconds = time.time()
        secondsElapsed = float(seconds - seconds_last)
        seconds_last = seconds
        remainingEps = (max_step + 1) - (step + 1)
        estRemainSecondsInt = int(secondsElapsed) * (remainingEps)
        print("Time for epoch: " + str(int(secondsElapsed)) + "seconds")

        print(
            "Est remaining time: "
            + str(datetime.timedelta(seconds=estRemainSecondsInt))
            + " or "
            + str(estRemainSecondsInt)
            + " seconds"
        )

        print("----------------------------------------", flush=True)
        step += 1
