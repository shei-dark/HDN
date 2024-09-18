import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
import random
from glob import glob
import os
import numpy as np
import torch
from tqdm import tqdm
from random import shuffle
import torch.nn.functional as F
import random

class Custom3DDataset(Dataset):
    """
    A custom dataset that extracts patches from 3D images and extract them based on their labels.
    """


    def __init__(self, images, labels, patch_size=(64,64,64)):
        """
        Initialize the Custom3DDataset by extracting valid 3D patches.

        Parameters:
        -----------
        images : dict
            A dictionary of 3D images.
        labels : dict
            A dictionary of corresponding labels for the 3D images.
        patch_size : tuple
            Size of the 3D patches (default is (64, 64, 64)).
        mask_size : tuple
            Size of the masked area in 3D (default is (5, 5, 5)).
        """
        self.patch_size = patch_size
        self.all_patches = []  # List to store all patches (with different labels)
        self.patches_by_label = self._extract_valid_patches(images, labels)


    def __len__(self):
        return len(self.all_patches)


    def _extract_valid_patches(self, images, labels):
        """
        Extracts valid 3D patches from the given images based on the provided labels.

        Parameters:
        -----------
        images : list
            A list of 3D images.
        labels : list
            A list of corresponding labels for the 3D images.

        Returns:
        --------
        patches_by_label : dict
            A dictionary mapping labels to indices of patches.
        """
        patches_by_label = {}
        for img, lbl in tqdm(zip(images, labels), f'Extracting patches from training data'):
            depth, height, width = img.shape
            d_patch, h_patch, w_patch = self.patch_size
            # Iterate over 3D volumes (depth, height, width) to extract patches
            for i in range(0, depth//d_patch):
                for j in range(0, height//h_patch):
                    for k in range(0, width//w_patch):
                        z = i * d_patch
                        y = j * h_patch
                        x = k * w_patch

                        # Extract 3D patch and corresponding label
                        patch = img[z : z + d_patch, y : y + h_patch, x : x + w_patch]
                        patch_label = lbl[z : z + d_patch, y : y + h_patch, x : x + w_patch]

                        # Extract blind spot area in the center
                        z_center = d_patch // 2
                        y_center = h_patch // 2
                        x_center = w_patch // 2

                        center_label = patch_label[z_center, y_center, x_center]

                        if center_label != -1:
                            if center_label not in patches_by_label:
                                patches_by_label[center_label] = []
                            
                            # Store patch, center label, and full patch label
                            self.all_patches.append(
                                (
                                    torch.tensor(patch).unsqueeze(0),  # Add channel dimension
                                    torch.tensor(center_label),
                                    torch.tensor(patch_label).unsqueeze(0)
                                )
                            )
                            patches_by_label[center_label].append(len(self.all_patches) - 1)
        return patches_by_label