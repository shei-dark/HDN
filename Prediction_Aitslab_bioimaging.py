import torch
import numpy as np
from tqdm import tqdm
from boilerplate.dataloader import CustomTestDataset
import tifffile as tiff
from torch.utils.data import DataLoader
import scipy.ndimage as ndi

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

hierarchy_level = 3
data_dir = "/group/jug/Sheida/Aitslab_bioimaging/img/test/"
key = [
    "cell_2_nuclei_0.tif",
    "cell_3_nuclei_1.tif",
    "cell_4_nuclei_2.tif",
    "cell_5_nuclei_3.tif",
    "cell_6_nuclei_4.tif",
    "cell_7_nuclei_5.tif",
    "cell_8_nuclei_6.tif",
    "cell_9_nuclei_7.tif",
    "cell_10_nuclei_8.tif",
    "cell_11_nuclei_9.tif",]

for k in key:
    test_images = tiff.imread(data_dir + k)

    model_dir = "/group/jug/Sheida/HVAE/experiments/"
    model_versions = ["25"]
    batch_size = 1024

    print("Processing test dataset")
    test_dataset = CustomTestDataset(
        test_images, patch_size=(2, 64, 64), stride=1, model="2D_multichannel"
    )
    print("Test dataset loaded. Processing test dataloader")
    dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    for model_v in model_versions:
        model = torch.load(model_dir + model_v + "/model/experiments_best_vae.net")
        data_mean = model.data_mean
        data_std = model.data_std
        model.mode_pred = True
        model.eval()
        device = model.device
        print(f"Processing image {k} with model version {model_v}")
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
                labels = output['pi'].argmax(dim=1)
                pred.extend(labels.cpu().numpy())

        pred_array = np.array(pred)

        clusters = pred_array.reshape(
            test_dataset.num_patches_y, test_dataset.num_patches_x
        )
        tiff.imwrite(f"{model_dir}{model_v}/seg/{k}_pred_reverse.tif", clusters.astype(np.uint8))
        print(f"Segmentation for image slice {k} pred saved")