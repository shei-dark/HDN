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
prediction_dir = "/group/jug/Sheida/output/high_c4_classifier3.tif"
gt_stack_path = "/group/jug/Sheida/pancreatic beta cells/download/high_c4/high_c4_gt.tif"   # Path to the ground truth stack
pred_stack = tiff.imread(prediction_dir).astype(np.int16)  # **Ensure predictions are int16**
pred_stack = pred_stack[49:1016]  # Adjust range if necessary
# Load ground truth stack
gt_stack = tiff.imread(gt_stack_path).astype(np.int16)  # **Ensure GT is int16**
gt_stack = gt_stack[49:1016]  # Adjust range if necessary

# Get unique classes from GT (including 0, 1, 2, 3 but excluding -1)
num_classes = int(np.nanmax(gt_stack) + 1)  # Avoid using -1

# **Get shape dynamically**
_, pred_height, pred_width = pred_stack.shape  # **Dynamically determine correct shape**

    # **Mask `-1` pixels in GT AFTER cropping**
mask_outside = gt_stack == -1

    # Preprocess prediction with noise removal
# pred_processed = preprocess_prediction(pred_stack, noise_threshold=10, dilation_size=1)

# **Now, apply the `-1` mask from GT to Predictions**
pred_stack[mask_outside] = -1
pred_stack[pred_stack == 2] = 4
pred_stack[pred_stack == 1] = 5
pred_stack[pred_stack == 0] = 6
pred_stack[pred_stack == 4] = 0
pred_stack[pred_stack == 5] = 2
pred_stack[pred_stack == 6] = 1
# pred_processed[mask_outside] = -1

   
# Compute Dice across the entire stack, ignoring -1
dice_before = compute_dice_score(pred_stack, gt_stack, num_classes)
# dice_after = compute_dice_score(pred_processed, gt_stack, num_classes)


# Compute mean DSC (excluding NaNs)
mean_dice_before = np.nanmean(list(dice_before.values()))
# mean_dice_after = np.nanmean(list(dice_after.values()))

# Print results
print("Per-Class Dice Scores (Before Preprocessing):")
for cls, score in dice_before.items():
    if not np.isnan(score):  # Ignore classes that were completely missing
        print(f"Class {cls}: {score:.4f}")
print(f"\nMean DSC Before: {mean_dice_before:.4f}")

# print("\nPer-Class Dice Scores (After Preprocessing):")
# for cls, score in dice_after.items():
    # if not np.isnan(score):  # Ignore classes that were completely missing
        # print(f"Class {cls}: {score:.4f}")
# print(f"\nMean DSC After: {mean_dice_after:.4f}")

