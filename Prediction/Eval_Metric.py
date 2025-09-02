import numpy as np
import tifffile as tiff
from sklearn.metrics import precision_recall_curve, auc

# Load processed predictions
processed_pred_path = "/group/jug/Sheida/HVAE/segmentation/03/processed_predictions.tif"  # Update path if needed
processed_pred_stack = tiff.imread(processed_pred_path).astype(np.int16)

# Load ground truth
gt_stack_path = "/group/jug/Sheida/pancreatic beta cells/download/high_c4/high_c4_gt.tif"  # Update path
gt_stack = tiff.imread(gt_stack_path).astype(np.int16)
gt_stack = gt_stack[49:1016]  # Adjust if needed

# Ensure GT is cropped to match the size of processed predictions
assert len(gt_stack) == len(processed_pred_stack), "Mismatch in number of slices!"

# Get prediction dimensions
pred_height, pred_width = processed_pred_stack.shape[1], processed_pred_stack.shape[2]

# Crop GT to center to match the processed prediction size
cropped_gt_stack = np.zeros_like(processed_pred_stack, dtype=np.int16)
for i in range(len(gt_stack)):
    gt_slice = gt_stack[i]  # Full GT slice
    start_x = (gt_slice.shape[1] - pred_width) // 2  # Center crop X
    start_y = (gt_slice.shape[0] - pred_height) // 2  # Center crop Y
    cropped_gt_stack[i] = gt_slice[
        start_y : start_y + pred_height, start_x : start_x + pred_width
    ]

# Ensure final cropped GT matches prediction dimensions
assert (
    cropped_gt_stack.shape == processed_pred_stack.shape
), "GT cropping failed, mismatched shape!"

# Get unique classes (excluding -1)
num_classes = int(np.nanmax(cropped_gt_stack) + 1)
valid_classes = [cls for cls in range(num_classes) if cls != -1]

# Initialize metric storage
iou_scores = {}
precision_scores = {}
recall_scores = {}
mAP_scores = {}

for cls in valid_classes:
    # Mask valid pixels (ignore -1)
    valid_mask = cropped_gt_stack != -1

    # Compute TP, FP, FN
    pred_bin = (processed_pred_stack == cls) & valid_mask
    gt_bin = (cropped_gt_stack == cls) & valid_mask

    intersection = np.sum(pred_bin & gt_bin)
    union = np.sum(pred_bin | gt_bin)

    TP = intersection
    FP = np.sum(pred_bin) - TP
    FN = np.sum(gt_bin) - TP

    # Compute IoU
    iou_scores[cls] = TP / union if union > 0 else np.nan

    # Compute Precision & Recall
    precision_scores[cls] = TP / (TP + FP) if (TP + FP) > 0 else np.nan
    recall_scores[cls] = TP / (TP + FN) if (TP + FN) > 0 else np.nan

    # Compute Precision-Recall curve for mAP
    precision, recall, _ = precision_recall_curve(gt_bin.flatten(), pred_bin.flatten())
    mAP_scores[cls] = auc(recall, precision)

# Compute Mean IoU, Mean Precision, Mean Recall, Mean mAP
mean_iou = np.nanmean(list(iou_scores.values()))
mean_precision = np.nanmean(list(precision_scores.values()))
mean_recall = np.nanmean(list(recall_scores.values()))
mean_mAP = np.nanmean(list(mAP_scores.values()))

# Print Results
print("\nEvaluation Metrics Per Class:")
for cls in valid_classes:
    print(
        f"Class {cls}: IoU={iou_scores[cls]:.4f}, Precision={precision_scores[cls]:.4f}, Recall={recall_scores[cls]:.4f}, mAP={mAP_scores[cls]:.4f}"
    )

print("\nOverall Performance:")
print(f"Mean IoU: {mean_iou:.4f}")
print(f"Mean Precision: {mean_precision:.4f}")
print(f"Mean Recall: {mean_recall:.4f}")
print(f"Mean mAP: {mean_mAP:.4f}")
