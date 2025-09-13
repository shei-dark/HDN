import os, sys
HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..'))  # parent of myscript/
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import time
import torch
import numpy as np

# from lib.dataloader import CustomTestDataset
from boilerplate.dataloader import CustomTestDataset, NonNeg1CenterPatchDataset
import tifffile as tiff
import os
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

batch_size = 1024
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
model_dir = "/group/jug/Sheida/HVAE/WACV/"
img_idx = range(49,1016)
# img_idx = [626]
model_versions = ["epsSeg"]
batch_size = 1024

for test_index in img_idx:
    print("Processing test dataset")

    # DATASET that only yields valid centers (label != -1)
    test_dataset = NonNeg1CenterPatchDataset(
        image=test_images, label=test_ground_truth_image, z=test_index, patch_size=64
    )

    print(f"Valid centers for z={test_index}: {len(test_dataset)}")
    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    for model_v in model_versions:
        model_path = os.path.join(model_dir, model_v, "model_supervised", "best.net")
        model = torch.load(model_path, weights_only=False)
        model.eval()

        device = next(model.parameters()).device if hasattr(model, "parameters") else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data_mean = getattr(model, "data_mean", 0.0)
        data_std  = getattr(model, "data_std", 1.0)

        print(f"Processing image slice {test_index} with model version {model_v}")

        # Prepare output map initialized with -1
        H, W = test_ground_truth_image.shape[1:]
        pred_slice = np.full((H, W), fill_value=-1, dtype=np.int16)

        with torch.no_grad():
            for batch in dataloader:
                patches = batch["patch"].to(device)               # (B,1,64,64)
                # normalize (broadcast-safe)
                patches = (patches - torch.as_tensor(data_mean, device=device)) / torch.as_tensor(data_std, device=device)
                model.eval()
                model.mode_pred = True
                output = model(patches)                           # expects dict with "pi"
                y_pred = output["pi"].argmax(dim=-1).cpu().numpy().astype(np.int16)  # (B,)

                ys = batch["y"].numpy()
                xs = batch["x"].numpy()
                pred_slice[ys, xs] = y_pred                       # scatter center predictions

        # Save per-slice prediction; ignored pixels remain -1
        out_dir = f"/group/jug/Sheida/HVAE/WACV/{model_v}/seg/"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{test_index}_sup.tif")
        tiff.imwrite(out_path, pred_slice.astype(np.int8))
        print(f"Segmentation for image slice {test_index} saved to {out_path}")
