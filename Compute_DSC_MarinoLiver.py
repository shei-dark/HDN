import numpy as np
import tifffile as tiff
import os
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

    for cls in range(num_classes):
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

# Paths to datasets
gt_paths = [
    "/facility/imganfacusers/Sheida/combined_single_label/crop_00/labs.tif",
    "/facility/imganfacusers/Sheida/combined_single_label/crop_10/labs.tif"
]

pred_dirs = [
    "/group/jug/Sheida/HVAE/segmentation/19/seg_supervised/crop_00/",
    "/group/jug/Sheida/HVAE/segmentation/19/seg_supervised/crop_10/"
]
pred_dirs = "/facility/imganfacusers/Sheida/Zerial_unet/full_unet/output/zerial_unet_2D_large/results/zerial_unet_2D_large_1/per_image_binarized/"

# Load and concatenate ground truth stacks
gt_stacks = [tiff.imread(path).astype(np.int16) for path in gt_paths]
gt_stack = np.concatenate(gt_stacks, axis=0)  # Concatenate along z-axis

# Get list of prediction files for both volumes
pred_files = sorted(glob(os.path.join(pred_dirs, "*.tif")), 
                    key=lambda x: (int(os.path.basename(x).split('_')[1]),  # Primary sort key (e.g., 00, 10)
                                   int(os.path.basename(x).split('_')[2].split('.')[0])))  # Secondary sort key (e.g., 0, 127)

# pred_files = sum(pred_files, [])  # Flatten list

# Ensure number of slices match
assert len(pred_files) == gt_stack.shape[0], "Mismatch in number of prediction and ground truth images!"

# Get unique classes from GT (excluding -1)
num_classes = int(np.nanmax(gt_stack) + 1)

# **Get shape dynamically**
first_pred = tiff.imread(pred_files[0]).astype(np.int16)
pred_height, pred_width = first_pred.shape

# Initialize stacks
full_pred_stack = np.zeros((len(pred_files), pred_height, pred_width), dtype=np.int16)
# full_pred_stack_processed = np.zeros((len(pred_files), pred_height, pred_width), dtype=np.int16)
full_gt_stack = np.zeros((len(pred_files), pred_height, pred_width), dtype=np.int16)

for idx, pred_path in enumerate(pred_files):
    # Load prediction and convert to int16
    pred = tiff.imread(pred_path).astype(np.int16)

    # Ensure the prediction matches expected shape
    if pred.shape != (pred_height, pred_width):
        print(f"Warning: Resizing prediction {idx} from {pred.shape} to {(pred_height, pred_width)}")
        pred = np.resize(pred, (pred_height, pred_width))

    # Extract corresponding GT image
    gt_full = gt_stack[idx]
    start_x = (gt_full.shape[1] - pred_width) // 2
    start_y = (gt_full.shape[0] - pred_height) // 2
    gt_cropped = gt_full#[start_y:start_y + pred_height, start_x:start_x + pred_width]

    # **Mask `-1` pixels in GT AFTER cropping**
    mask_outside = gt_cropped == -1

    # Preprocess prediction
    # pred_processed = preprocess_prediction(pred, noise_threshold=10, dilation_size=1)
    
    # Apply `-1` mask
    pred[mask_outside] = -1
    # pred_processed[mask_outside] = -1

    # Store images in stacks
    full_pred_stack[idx] = pred
    # full_pred_stack_processed[idx] = pred_processed
    full_gt_stack[idx] = gt_cropped

# Compute Dice scores
dice_before = compute_dice_score(full_pred_stack, full_gt_stack, num_classes)
# dice_after = compute_dice_score(full_pred_stack_processed, full_gt_stack, num_classes)
# tiff.imwrite("/group/jug/Sheida/HVAE/segmentation/19/processed_predictions.tif", full_pred_stack_processed.astype(np.int16))
# Compute mean DSC (excluding NaNs)
mean_dice_before = np.nanmean(list(dice_before.values()))
# mean_dice_after = np.nanmean(list(dice_after.values()))

# Print results
print("Per-Class Dice Scores (Before Preprocessing):")
for cls, score in dice_before.items():
    if not np.isnan(score):
        print(f"Class {cls}: {score:.4f}")
print(f"\nMean DSC Before: {mean_dice_before:.4f}")

# print("\nPer-Class Dice Scores (After Preprocessing):")
# for cls, score in dice_after.items():
#     if not np.isnan(score):
#         print(f"Class {cls}: {score:.4f}")
# print(f"\nMean DSC After: {mean_dice_after:.4f}")
