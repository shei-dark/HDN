import os, sys
HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..'))  # parent of myscript/
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import torch
import numpy as np
from tqdm import tqdm
from boilerplate.dataloader import CustomTestDataset
import tifffile as tiff
from torch.utils.data import DataLoader
import time
import datetime
from torch.amp import autocast



use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
cell_mean = 29.797266
cell_std = 31.11202
nuclei_mean = 19.316538
nuclei_std = 32.213627
batch = torch.randn(2048, 2, 128, 128)  # Example batch with 2 channels

# Create tensors for mean and std
channel_means = torch.tensor([cell_mean, nuclei_mean]).to(device=device)  # Shape: (C,)
channel_stds = torch.tensor([cell_std, nuclei_std]).to(device=device)     # Shape: (C,)

data_dir = "/group/jug/Sheida/Aitslab_bioimaging/img/test/"
key = [
    # "cell_2_nuclei_0.tif",
    # "cell_3_nuclei_1.tif",
    "cell_4_nuclei_2.tif",
    # "cell_5_nuclei_3.tif",
    # "cell_6_nuclei_4.tif",
    # "cell_7_nuclei_5.tif",
    # "cell_8_nuclei_6.tif",
    # "cell_9_nuclei_7.tif",
    # "cell_10_nuclei_8.tif",
    # "cell_11_nuclei_9.tif",
    ]

model_dir = "/group/jug/Sheida/HVAE/segmentation/"
model_v = "21"
batch_size = 2048

max_step = len(key) 
step = 0 
seconds_last = time.time()
model = torch.load(model_dir + model_v + "/model_supervised/experiments_best_vae.net")
model.mode_pred = True
model.eval()
device = model.device
for k in key:
    test_images = tiff.imread(data_dir + k)

    print("Processing test dataset")
    test_dataset = CustomTestDataset(
        test_images, patch_size=(2, 64, 64), stride=1, model="2D_multichannel"
    )
    print("Test dataset loaded. Processing test dataloader")
    dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True
    )
    
    print(f"Processing image {k} with model version {model_v}")
    index = 0

    pred = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device, non_blocking=True)
            batch = batch.float()
            batch.sub_(channel_means.view(-1, 1, 1)).div_(channel_stds.view(-1, 1, 1))
            with autocast(device_type='cuda'):  # Enable mixed precision
                    output = model(batch)
            labels = output['pi'].argmax(dim=1)
            pred.extend(labels.cpu().numpy())

    pred_array = np.array(pred)

    clusters = pred_array.reshape(
        test_dataset.num_patches_y, test_dataset.num_patches_x
    )
    seg_dir = f"{model_dir}/{model_v}/seg_softmax/" #TODO
    os.makedirs(seg_dir, exist_ok=True)
    tiff.imwrite(f"{seg_dir}{k}.tif", clusters.astype(np.uint8))
    print(
        f"Segmentation for image slice {k} with model {model_v} is saved"
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