import os, sys
HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..'))  # parent of myscript/
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import numpy as np
import tifffile as tiff
from glob import glob
from skimage.morphology import binary_dilation, binary_erosion, remove_small_objects, disk
from scipy.ndimage import median_filter

def preprocess_prediction(pred, noise_threshold=5, dilation_size=1):
    """
    Preprocess segmentation by:
    - Removing small noise (single pixels)
    - Using median filtering for rare classes instead of erosion
    - Applying gentle morphological smoothing
    """
    unique_classes = np.unique(pred)
    pred_processed = np.zeros_like(pred)

    for cls in unique_classes:
        if cls == -1:  # Ignore -1 (background/outside)
            continue

        # Extract only current class
        class_mask = (pred == cls)

        # Remove small isolated noise pixels
        class_mask_cleaned = remove_small_objects(class_mask, min_size=noise_threshold)

        # Apply median filtering for slight smoothing
        class_mask_filtered = median_filter(class_mask_cleaned.astype(np.uint8), size=3)

        # Use dilation and erosion for boundary refinement
        selem = disk(dilation_size)
        class_mask_dilated = binary_dilation(class_mask_filtered, selem)
        class_mask_processed = binary_erosion(class_mask_dilated, selem)

        # Merge results
        pred_processed[class_mask_processed] = cls

    return pred_processed.astype(np.int16)  # Store as int16 to support -1

def compute_dice_score(pred_stack, gt_stack, num_classes):
    """Compute overall Dice coefficient per class for the entire stack, ignoring -1 in GT."""
    dice_scores = {}

    for cls in range(num_classes):  # **Include class 0**
        if cls == -1:  # Skip -1 (outside)
            continue
        
        # Ignore pixels where GT is -1
        valid_mask = gt_stack != -1

        pred_bin = (pred_stack == cls) & valid_mask
        gt_bin = (gt_stack == cls) & valid_mask
        
        intersection = np.sum(pred_bin & gt_bin)
        denominator = np.sum(pred_bin) + np.sum(gt_bin)
        
        if denominator == 0:
            dice_scores[cls] = np.nan  # Ignore if class is missing in both pred & GT
        else:
            dice_scores[cls] = (2.0 * intersection) / denominator
            
    return dice_scores

# Path to files (Modify accordingly)
prediction_dir = "/group/jug/Sheida/HVAE/segmentation/40/"  # Folder containing 0.tif, 1.tif, ..., 127.tif
# prediction_dir = "/facility/imganfacusers/Sheida/pancreatic_beta_cells/masked/2D/output/pancreas_unet/results/pancreas_unet_1/per_image_binarized/"
# prediction_dir = "/facility/imganfacusers/Sheida/pancreatic_beta_cells/masked/2D/small_unet/inference/pancreas_unet/results/pancreas_unet_1/per_image_binarized/"
gt_stack_path = "/group/jug/Sheida/pancreatic beta cells/download/high_c4/high_c4_gt.tif"   # Path to the ground truth stack
# output_path = f"{prediction_dir[:-19]}processed_predictions_supervised.tif"  # Output file for processed predictions

# Load ground truth stack
gt_stack = tiff.imread(gt_stack_path).astype(np.int16)  # **Ensure GT is int16**
gt_stack = gt_stack[626]  # Adjust range if necessary

# Get list of prediction files
# pred_files = sorted(glob(os.path.join(prediction_dir, "*.tif")), key=lambda x: int(os.path.basename(x).split('.')[0]))
# pred_files = sorted(glob(os.path.join(prediction_dir, "*.tif")), key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
pred_files = os.path.join(prediction_dir, "seg/626_semisup.tif")
# Ensure we have the right number of images
# assert len(pred_files) == gt_stack.shape[0], "Mismatch in number of prediction and ground truth images!"

# Get unique classes from GT (including 0, 1, 2, 3 but excluding -1)
num_classes = int(np.nanmax(gt_stack) + 1)  # Avoid using -1

# **Get shape dynamically**
first_pred = tiff.imread(pred_files).astype(np.int16)  # **Convert to int16**
pred_height, pred_width = first_pred.shape  # **Dynamically determine correct shape**

# Initialize stacks with correct shape
full_pred_stack = np.zeros((len(pred_files), pred_height, pred_width), dtype=np.int16)  # Use int16 for -1
# full_pred_stack_processed = np.zeros((len(pred_files), pred_height, pred_width), dtype=np.int16)
full_gt_stack = np.zeros((len(pred_files), pred_height, pred_width), dtype=np.int16)  # Keep -1 values

# for idx, pred_path in enumerate(pred_files):
    # Load prediction and convert to int16
    # pred = tiff.imread(pred_path).astype(np.int16)  # **Convert to int16**
    # pred[pred==3] = 4
    # pred[pred==2] = 5
    # pred[pred==4] = 2
    # pred[pred==5] = 3
    # Ensure the prediction matches expected shape
    # if pred.shape != (pred_height, pred_width):
    #     print(f"Warning: Resizing prediction {idx} from {pred.shape} to {(pred_height, pred_width)}")
    #     pred = np.resize(pred, (pred_height, pred_width))  # Resize dynamically if needed

    # Extract corresponding GT image
gt_full = gt_stack#[idx]  # Shape: (1019, 482) from GT stack

    # **Crop GT first, ensuring alignment with prediction**
start_x = (gt_full.shape[1] - pred_width) // 2  # Center crop
start_y = (gt_full.shape[0] - pred_height) // 2
gt_cropped = gt_full[start_y:start_y + pred_height, start_x:start_x + pred_width]

    # **Mask `-1` pixels in GT AFTER cropping**
mask_outside = gt_cropped == -1

    # Preprocess prediction with noise removal
    # pred_processed = preprocess_prediction(pred, noise_threshold=10, dilation_size=1)

    # **Now, apply the `-1` mask from GT to Predictions**
first_pred[mask_outside] = -1
    # pred_processed[mask_outside] = -1

    # Store images in stacks
# full_pred_stack[idx] = pred  # Raw prediction with `-1`
    # full_pred_stack_processed[idx] = pred_processed  # Processed prediction with `-1`
# full_gt_stack[idx] = gt_cropped  # GT matched to prediction

# Compute Dice across the entire stack, ignoring -1
# dice_before = compute_dice_score(full_pred_stack, full_gt_stack, num_classes)
# dice_after = compute_dice_score(full_pred_stack_processed, full_gt_stack, num_classes)
dice = compute_dice_score(first_pred, gt_cropped, num_classes)
# Save processed predictions as a multi-page TIFF file
# tiff.imwrite(output_path, full_pred_stack_processed, dtype=np.int16)

# Compute mean DSC (excluding NaNs)
mean_dice_before = np.nanmean(list(dice.values()))
# mean_dice_after = np.nanmean(list(dice_after.values()))

# Print results
print("Per-Class Dice Scores (Before Preprocessing):")
for cls, score in dice.items():
    if not np.isnan(score):  # Ignore classes that were completely missing
        print(f"Class {cls}: {score:.4f}")
print(f"\nMean DSC Before: {mean_dice_before:.4f}")

# print("\nPer-Class Dice Scores (After Preprocessing):")
# for cls, score in dice_after.items():
#     if not np.isnan(score):  # Ignore classes that were completely missing
#         print(f"Class {cls}: {score:.4f}")
# print(f"\nMean DSC After: {mean_dice_after:.4f}")

# print(f"\nProcessed predictions saved to: {output_path}")
