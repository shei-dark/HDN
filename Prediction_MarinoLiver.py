import torch
import numpy as np
from tqdm import tqdm
from boilerplate.dataloader import CustomTestDataset
import tifffile as tiff
from torch.utils.data import DataLoader
import scipy.ndimage as ndi



use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu") 

dist_metric = ["cosine"]

# num_clusters = 5
# patch_size = (3, 64, 64)
hierarchy_level = 3
data_dir = "/facility/imganfacusers/Sheida/combined_single_label/"
key = ["crop_00", "crop_10"]
for k in key:
    test_images = tiff.imread(data_dir + k + "/image.tif")
    test_ground_truth_image = tiff.imread(data_dir + k + "/labs.tif")

    # compute mean and std of the data

    model_dir = "/group/jug/Sheida/HVAE/experiments/"
    img_idx = range(128)
    model_versions = ["34"]
    batch_size = 1024


    for test_index in tqdm(img_idx):
        print("Processing test dataset")
        test_dataset = CustomTestDataset(
            test_images, patch_size=(64, 64), index=test_index, stride=1, model="2D"
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
                    labels = output['pi'].argmax(dim=1)
                    pred.extend(labels.cpu().numpy())

            pred_array = np.array(pred)

            segmentation_np = pred_array.reshape(
                test_dataset.num_patches_y, test_dataset.num_patches_x
            )
            processed_map = np.zeros_like(segmentation_np)
            structure = ndi.generate_binary_structure(2, 1)

            for class_id in range(1,5):
                # Binary mask for the current class
                binary_mask = segmentation_np == class_id
                # 1️⃣ Remove small noise (despeckling)
                despeckled_mask = ndi.binary_opening(binary_mask, structure=structure)
                # 2️⃣ Fill small holes inside objects
                hole_filled_mask = ndi.binary_fill_holes(despeckled_mask)
                # 3️⃣ Restore object integrity (prevents breaking)
                closed_mask = ndi.binary_closing(hole_filled_mask, structure=structure)
                # 4️⃣ (Optional) Slight dilation to recover object thickness
                dilated_mask = ndi.binary_dilation(closed_mask, structure=structure, iterations=1)
                eroded_mask = ndi.binary_erosion(dilated_mask, structure=structure, iterations=1)
                dilated_mask = ndi.binary_dilation(eroded_mask, structure=structure, iterations=1)
                eroded_mask = ndi.binary_erosion(dilated_mask, structure=structure, iterations=1)
                # Store processed mask
                processed_map[eroded_mask] = class_id
                
            tiff.imwrite(f"{model_dir}{model_v}/seg/{k}_{test_index}.tif", processed_map.astype(np.uint8))
            print(f"Segmentation for image slice {test_index} saved")