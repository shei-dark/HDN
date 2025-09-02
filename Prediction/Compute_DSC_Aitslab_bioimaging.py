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
gt_paths = []

# Additional dataset
gt_keys = [
    "cell_2_nuclei_0.tif", "cell_3_nuclei_1.tif", "cell_4_nuclei_2.tif", "cell_5_nuclei_3.tif", 
    "cell_6_nuclei_4.tif", "cell_7_nuclei_5.tif", "cell_8_nuclei_6.tif", "cell_9_nuclei_7.tif", 
    "cell_10_nuclei_8.tif", "cell_11_nuclei_9.tif"
]
gt_paths += [f"/group/jug/Sheida/Aitslab_bioimaging/gt/test/{k}" for k in gt_keys]

pred_paths = [f"/group/jug/Sheida/HVAE/segmentation/21/seg_supervised/{k}.tif" for k in gt_keys]

# Load and concatenate ground truth stacks
gt_stacks = []
for path in gt_paths:
    gt = tiff.imread(path).astype(np.int16)
    if gt.ndim == 3:
        gt = gt[0]  # Take the first slice if 3D
    # Crop center to match prediction size
    start_x = (gt.shape[1] - 1041) // 2
    start_y = (gt.shape[0] - 1041) // 2
    gt = gt[start_y:start_y + 1041, start_x:start_x + 1041]
    gt_stacks.append(gt)
gt_stack = np.stack(gt_stacks, axis=0)  # Stack along new axis

# Convert GT: Treat 1 and 3 as one class (both as 1)
gt_stack[gt_stack == 3] = 1

# Load and concatenate prediction stacks
pred_stacks = []
for path in pred_paths:
    pred = tiff.imread(path).astype(np.int16)
    if pred.ndim == 3:
        pred = pred[0]  # Take the first slice if 3D
    pred_stacks.append(pred)
pred_stack = np.stack(pred_stacks, axis=0)  # Stack along new axis

# Ensure number of slices match
assert pred_stack.shape == gt_stack.shape, f"Mismatch in number of prediction and ground truth images! Pred shape: {pred_stack.shape}, GT shape: {gt_stack.shape}"

# Get unique classes from GT (excluding -1)
num_classes = int(np.nanmax(gt_stack) + 1)

# Preprocess predictions
pred_stack_processed = np.array([preprocess_prediction(slice_) for slice_ in pred_stack])

# Save post-processed predictions
tiff.imwrite("/group/jug/Sheida/HVAE/segmentation/21/seg_supervised/processed_predictions.tif", pred_stack_processed.astype(np.int16))

# Compute Dice scores
dice_before = compute_dice_score(pred_stack, gt_stack, num_classes)
dice_after = compute_dice_score(pred_stack_processed, gt_stack, num_classes)

# Compute mean DSC (excluding NaNs)
mean_dice_before = np.nanmean(list(dice_before.values()))
mean_dice_after = np.nanmean(list(dice_after.values()))

# Print results
print("Per-Class Dice Scores (Before Preprocessing):")
for cls, score in dice_before.items():
    if not np.isnan(score):
        print(f"Class {cls}: {score:.4f}")
print(f"\nMean DSC Before: {mean_dice_before:.4f}")

print("\nPer-Class Dice Scores (After Preprocessing):")
for cls, score in dice_after.items():
    if not np.isnan(score):
        print(f"Class {cls}: {score:.4f}")
print(f"\nMean DSC After: {mean_dice_after:.4f}")
