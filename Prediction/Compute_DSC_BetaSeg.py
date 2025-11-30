import os
import sys
from glob import glob

import numpy as np
import tifffile as tiff

HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))  
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def center_crop_to(gt_img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Center-crop a 2D array to (target_h, target_w)."""
    gh, gw = gt_img.shape
    if gh < target_h or gw < target_w:
        raise ValueError(
            f"GT slice {gt_img.shape} is smaller than target {(target_h, target_w)}."
        )
    sy = (gh - target_h) // 2
    sx = (gw - target_w) // 2
    return gt_img[sy : sy + target_h, sx : sx + target_w]


def dice_per_class(pred_stack: np.ndarray, gt_stack: np.ndarray) -> dict:
    """
    Compute Dice per class over the entire stack, ignoring GT == -1.
    Returns {class_id: dice or np.nan}.
    """
    assert pred_stack.shape == gt_stack.shape, "pred and gt stacks must match in shape"
    valid = gt_stack != -1

    # Classes present in either pred or gt (excluding -1)
    classes = np.unique(np.concatenate([pred_stack[valid].ravel(), gt_stack[valid].ravel()]))
    classes = classes[classes >= 0]  # keep 0..K

    scores = {}
    for cls in classes:
        pred_bin = (pred_stack == cls) & valid
        gt_bin = (gt_stack == cls) & valid
        intersection = np.sum(pred_bin & gt_bin)
        denom = np.sum(pred_bin) + np.sum(gt_bin)
        scores[int(cls)] = np.nan if denom == 0 else (2.0 * intersection) / denom

    return scores

# /group/jug/Sheida/HVAE/plus/{model_v}/seg++/{test_index}.tif
def main():
    for model_name in ["08"]:
        prediction_dir = f"/group/jug/Sheida/HVAE/plus/{model_name}/seg++/626.tif"
        gt_stack_path = "/group/jug/Sheida/pancreatic beta cells/download/high_c4/high_c4_gt.tif"

        # Load GT (keep int16 to allow -1)
        gt_full_stack = tiff.imread(gt_stack_path).astype(np.int16)
        gt_full_stack = gt_full_stack[626:627]#[49:1016]  # adjust if needed
        n_slices_gt = gt_full_stack.shape[0]

        # Collect prediction files (sorted numerically by basename without extension)
        # pred_files = sorted(
        #     glob(os.path.join(prediction_dir, "*.tif")),
        #     key=lambda p: int(os.path.basename(p).split(".")[0]),
        # )
        pred_files = glob(os.path.join(prediction_dir))
        if not pred_files:
            raise FileNotFoundError(f"No .tif predictions in: {prediction_dir}")

        if len(pred_files) != n_slices_gt:
            raise AssertionError(
                f"Mismatch: {len(pred_files)} pred slices vs {n_slices_gt} GT slices."
            )

        # Read first pred to lock shape
        first_pred = tiff.imread(pred_files[0]).astype(np.int16)
        pred_h, pred_w = first_pred.shape

        # Allocate stacks
        n = len(pred_files)
        pred_stack = np.zeros((n, pred_h, pred_w), dtype=np.int16)
        gt_stack = np.zeros((n, pred_h, pred_w), dtype=np.int16)

        # Fill stacks (align GT to pred via center-crop)
        for idx, pred_path in enumerate(pred_files):
            pred = tiff.imread(pred_path).astype(np.int16)
            if pred.shape != (pred_h, pred_w):
                raise ValueError(
                    f"Pred slice {pred_path} has shape {pred.shape}, expected {(pred_h, pred_w)}."
                )
            gt_slice = gt_full_stack[idx]
            gt_cropped = center_crop_to(gt_slice, pred_h, pred_w)

            pred_stack[idx] = pred
            gt_stack[idx] = gt_cropped

        # Compute Dice scores
        dice = dice_per_class(pred_stack, gt_stack)
        mean_dice = np.nanmean(list(dice.values())) if dice else np.nan

        # Report
        print("Per-class Dice:")
        for cls in sorted(dice.keys()):
            val = dice[cls]
            print(f"  Class {cls}: {val:.4f}" if not np.isnan(val) else f"  Class {cls}: NaN")

        print(f"\nMean DSC: {mean_dice:.4f}" if not np.isnan(mean_dice) else "\nMean DSC: NaN")


if __name__ == "__main__":
    main()
