import os, sys

HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))  # parent of myscript/
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    
import torch
import numpy as np
from boilerplate.dataloader import CustomTestDataset
import tifffile as tiff
from torch.utils.data import DataLoader
import time
import datetime
from torch.amp import autocast

# Fix device setting
device = "cuda" if torch.cuda.is_available() else "cpu"

with autocast(device_type=device):  # Ensure 'cuda' or 'cpu'
    print("Device: ", device)
    num_clusters = 7
    patch_size = (1, 64, 64)
    hierarchy_level = 3
    data_dir = "/facility/imganfacusers/Sheida/combined_single_label/"
    key = ["crop_00", "crop_10"]
    for k in key:
        test_images = tiff.imread(data_dir + k + "/image.tif")
        print("Test image loaded from path:")
        print(data_dir + k + "/image.tif")
        model_dir = "/group/jug/Sheida/HVAE/segmentation/"
        img_idx = [26]#range(128)
        model_v = "19"
        batch_size = 1024

        max_step = len(img_idx)
        step = 0
        seconds_last = time.time()
        model = torch.load(
            model_dir + model_v + "/model_supervised/experiments_best_vae.net",
            weights_only=False,
        )
        data_mean = model.data_mean
        data_std = model.data_std
        model.mode_pred = True
        model.eval().to(device)
        
        # FIX: Don't overwrite 'device' with model.device
        # device = model.device  # REMOVE THIS
        device = "cuda" if torch.cuda.is_available() else "cpu"  # Keep this

        for test_index in img_idx:
            print("Processing test dataset")
            test_dataset = CustomTestDataset(
                test_images, patch_size=(64, 64), index=test_index, stride=1, model="2D"
            )
            print("Test dataset loaded. Processing test dataloader")
            dataloader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True,
            )
            print(f"Processing image slice {test_index} with model version {model_v}")
            index = 0
            pred = []
            with torch.no_grad():
                for batch in dataloader:
                    batch = batch.to(device, non_blocking=True)
                    batch = batch.float()
                    batch.sub_(data_mean).div_(data_std)
                    with autocast(device_type=device):  # Ensure 'cuda' or 'cpu'
                        output = model(batch)
                    y_pred = output['pi'].argmax(dim=-1)
                    pred.extend(y_pred.cpu().numpy())

            pred_array = np.array(pred)

            segmentation_np = pred_array.reshape(
                test_dataset.num_patches_y, test_dataset.num_patches_x
            )
            seg_dir = f"{model_dir}{model_v}/seg_supervised_marino/{k}/"
            os.makedirs(seg_dir, exist_ok=True)
            tiff.imwrite(f"{seg_dir}{test_index}.tif", segmentation_np.astype(np.uint8))
            print(f"Segmentation for image slice {test_index} of image {k} with model {model_v} is saved")
            
            seconds = time.time()
            secondsElapsed = float(seconds - seconds_last)
            seconds_last = seconds
            remainingEps = max_step - step
            estRemainSecondsInt = int(secondsElapsed * remainingEps)
            print("Time for epoch:" + str(int(secondsElapsed)) + "seconds")
            print(f"Estimated remaining time: {datetime.timedelta(seconds=estRemainSecondsInt)} or {estRemainSecondsInt} seconds")
            print("-------------------", flush=True)
            step += 1
