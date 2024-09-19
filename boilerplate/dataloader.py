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


    def __init__(self, images, labels, patch_size=(64,64,64), mask_size=(5,5,5)):
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
        self.mask_size = mask_size
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
        for img, lbl in zip(images, labels):
            depth, height, width = img.shape
            d_patch, h_patch, w_patch = self.patch_size
            z_center = d_patch // 2
            y_center = h_patch // 2
            x_center = w_patch // 2
            d_stride, h_stride, w_stride = self.mask_size
            d_stride *= 3
            h_stride *= 3
            w_stride *= 3
            # Iterate over 3D volumes (depth, height, width) to extract patches
            for z in tqdm(range(0, depth, d_stride), f'Extracting patches from volume: '):
                for y in range(0, height, h_stride):
                    for x in range(0, width, w_stride):

                        # Extract 3D patch and corresponding label
                        patch = img[z : z + d_patch, y : y + h_patch, x : x + w_patch]
                        patch_label = lbl[z : z + d_patch, y : y + h_patch, x : x + w_patch]
                        if patch.shape != self.patch_size:
                            continue
                        # Extract blind spot area in the center
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
    

    def __getitem__(self, idx):
        """
        Retrieves the 3D patch, its class, and the label map.

        Parameters:
        -----------
        idx : int
            Index of the patch.

        Returns:
        --------
        patch : torch.Tensor
            The 3D patch extracted from the image.
        cls : torch.Tensor
            The label of the patch's center.
        label : torch.Tensor
            The full label map of the patch.
        """
        patch, cls, label = self.all_patches[idx]
        return patch, cls, label
    

class CombinedCustom3DDataset(Custom3DDataset):
    """
    A combined dataset class that handles labeled and unlabeled 3D data for training with 
    contrastive loss and other unsupervised losses.
    """

    def __init__(self, images, labels, labeled_indices, patch_size=(64,64,64), mask_size=(5,5,5)):
        """
        Initialize the CombinedCustom3DDataset, separating patches into labeled and unlabeled subsets.

        Parameters:
        -----------
        images : dict
            A dictionary of 3D images.
        labels : dict
            A dictionary of corresponding labels for the 3D images.
        labeled_indices : list
            A list of indices of `all_patches` that should keep their labels for contrastive loss.
        patch_size : tuple
            Size of the 3D patches (default is (64, 64, 64)).
        mask_size : tuple
            Size of the masked area in 3D (default is (5, 5, 5)).
        """
        super().__init__(images, labels, patch_size, mask_size)
        
        # Separate labeled and unlabeled patches based on labeled_indices
        self.labeled_indices = labeled_indices
        self.random_patches = []
        self._get_random_patch(images, labels)
        self._update_patches_by_label()


    def __getitem__(self, idx):
        """
        Retrieves the patch, its class, and the label map.

        Parameters:
        -----------
        idx : int
            Index of the patch.

        Returns:
        --------
        patch : torch.Tensor
            The 3D patch extracted from the image.
        cls : torch.Tensor
            The label of the patch's center (-2 if the patch is from the unlabeled subset).
        label : torch.Tensor
            The full label map of the patch.
        """
        
        # Return with label set to -2 if the index is not part of labeled indices
        if isinstance(idx, list):
            patches = [self.all_patches[i] if i in self.labeled_indices else self.random_patches[i] for i in idx]
            patches, clss, labels = zip(*patches)
            return torch.stack(patches), torch.tensor(clss), torch.stack(labels)
        else:
            if idx in self.labeled_indices:
                patch, cls, label = self.all_patches[idx]
            else:
                patch, cls, label = self.random_patches[idx]
            return patch, cls, label

    def _get_random_patch(self, images, labels):
        """
        Retrieves a random patch from the dataset.

        Returns:
        --------
        patch : torch.Tensor
            The random 3D patch.
        label : torch.Tensor
            The label map of the random 3D patch.
        """
        keys = list(images.keys())
        for _ in range(len(self.all_patches)):
            key = random.choice(keys)
            img = images[key]
            lbl = labels[key]
            depth, height, width = img.shape

            z = random.randrange(0, depth - self.patch_size[0])
            y = random.randrange(0, height - self.patch_size[1])
            x = random.randrange(0, width - self.patch_size[2])

            patch = img[z : z + self.patch_size[0], y : y + self.patch_size[1], x : x + self.patch_size[2]]
            patch_label = lbl[z : z + self.patch_size[0], y : y + self.patch_size[1], x : x + self.patch_size[2]]

            self.random_patches.append((
                torch.tensor(patch).unsqueeze(0),  # Add channel dimension
                torch.tensor(-2),  # Label set to -2 for unlabeled
                torch.tensor(patch_label).unsqueeze(0)  # Add channel dimension
            ))
        return
    
    def _update_patches_by_label(self):
        """
        Updates patches_by_label dictionary to include only labeled patches.
        """
        for key in self.patches_by_label:
            self.patches_by_label[key] = [value for value in self.patches_by_label[key] if value in self.labeled_indices]
