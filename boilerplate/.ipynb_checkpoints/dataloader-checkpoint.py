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
                                    torch.tensor(patch_label).unsqueeze(0)  # Add channel dimension
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
        if isinstance(idx, list):
            patches = [self.all_patches[i] for i in idx]
            patches, clss, labels = zip(*patches)
            return torch.stack(patches).squeeze(0), torch.tensor(clss), torch.stack(labels)
        else:
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
        indices = range(len(images))
        for _ in tqdm(range(len(self.all_patches))):
            idx = random.choice(indices)
            img = images[idx]
            lbl = labels[idx]
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


class BalancedBatchSampler(Sampler):

    """
    A custom sampler that generates balanced batches from a dataset by ensuring each batch
    contains a balanced number of samples from each label class.

    This sampler is useful when training models with imbalanced datasets, as it helps to
    maintain an equal representation of each class within each batch. The class ensures
    that samples from each label are included in the batch proportionally and handles
    scenarios where the number of samples for each label differs significantly.

    Attributes:
    -----------
    dataset : Dataset
        The dataset from which samples are drawn. The dataset should have a `patches_by_label`
        attribute, which is a dictionary mapping labels to indices of samples belonging to
        those labels.
        
    batch_size : int
        The total number of samples in each batch.

    label_to_indices : dict
        A dictionary that maps each label to a list of indices of samples that belong
        to that label.

    num_labels : int
        The number of unique labels in the dataset.

    samples_per_label : int
        The number of samples to include from each label in each batch.

    remaining_samples : int
        The number of extra samples to distribute across labels to fill the batch.

    max_batch : int
        The maximum number of batches that can be generated based on the size of the
        largest class and the number of samples per label.

    Methods:
    --------
    __init__(dataset, batch_size)
        Initializes the sampler with the dataset and batch size.

    __iter__()
        Returns an iterator that yields balanced batches of indices.

    __len__()
        Estimates the total number of batches that can be generated.
       pass
    """

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

        # dictionary mapping labels to indices
        self.label_to_indices = dataset.patches_by_label
        for key in self.label_to_indices:
            shuffle(self.label_to_indices[key])
    
        # Determine number of labels
        self.num_labels = len(self.label_to_indices)
        self.samples_per_label = self.batch_size // self.num_labels
        self.remaining_samples = self.batch_size % self.num_labels  
        self.max_batch = max(len(indices) for indices in self.label_to_indices.values()) // self.samples_per_label

    def __iter__(self):
        max_class_size = max(len(indices) for indices in self.label_to_indices.values())
        num_batches_generated = 0
        # Generate balanced batches
        while num_batches_generated < self.max_batch:
            batch = []
            for label, indices in self.label_to_indices.items():
                if len(indices) < self.samples_per_label:
                    indices = random.choices(indices, k=max_class_size)  # Oversample
                selected_indices = random.sample(indices, self.samples_per_label)
                batch.extend(selected_indices)
            
            if len(batch) < self.batch_size:
                # Handle any remaining spots in the batch
                remaining_indices = []
                for indices in self.label_to_indices.values():
                    remaining_indices.extend(indices)
                random.shuffle(remaining_indices)
                batch.extend(remaining_indices[:self.batch_size - len(batch)])

            if len(batch) == self.batch_size:
                random.shuffle(batch)
                num_batches_generated += 1
                yield batch 
             # Return the batch and pause execution until the next batch is requested

    def __len__(self):
        # Estimate the length based on the largest class
        max_class_size = max(len(indices) for indices in self.label_to_indices.values())
        return (max_class_size * self.num_labels) // self.batch_size


class CombinedBatchSampler(Sampler):
    """
    A custom sampler that generates batches containing 25% balanced labeled samples
    and 75% random samples. Inherits from BalancedBatchSampler.
    """

    def __init__(self, dataset, batch_size, labeled_ratio=0.25):
        """
        Initializes the CombinedBatchSampler.

        Parameters:
        -----------
        dataset : Dataset
            The dataset from which samples are drawn. Should have a `patches_by_label`
            attribute for labeled patches.
        batch_size : int
            The total number of samples in each batch.
        labeled_indices : list
            List of indices of the labeled patches that should be used for contrastive loss.
        """
        self.label_to_indices = dataset.patches_by_label
        self.random_indices = [i for i in range(len(dataset)) if i not in dataset.labeled_indices]
        for key in self.label_to_indices:
            shuffle(self.label_to_indices[key])
        self.batch_size = batch_size
        self.small_batch_size = int(batch_size * labeled_ratio)
        self.num_labels = len(self.label_to_indices)
        self.samples_per_label = self.small_batch_size // self.num_labels
        self.remaining_samples = self.small_batch_size % self.num_labels
        self.max_batch = max(len(indices) for indices in self.label_to_indices.values()) // self.samples_per_label


    def __iter__(self):
        max_class_size = max(len(indices) for indices in self.label_to_indices.values())
        num_batches_generated = 0

        while num_batches_generated < self.max_batch:
            # Step 1: Sample 25% of the batch using balanced sampling from labeled indices
            balanced_batch = []
            for label, indices in self.label_to_indices.items():
                if len(indices) < self.samples_per_label:
                    indices = random.choices(indices, k=max_class_size)
                selected_indices = random.sample(indices, self.samples_per_label)
                balanced_batch.extend(selected_indices)

            # Fill up if the balanced batch is not full (due to class imbalance or fewer labeled samples)
            while len(balanced_batch) < self.small_batch_size:
                remaining_labeled = []
                for indices in self.label_to_indices.values():
                    remaining_labeled.extend(indices)
                random.shuffle(remaining_labeled)
                balanced_batch.extend(remaining_labeled[:self.small_batch_size - len(balanced_batch)])

            # Step 2: Sample 75% of the batch randomly from unlabeled indices
            random_unlabeled = random.sample(self.random_indices, self.batch_size - self.small_batch_size)

            # Combine balanced labeled and random unlabeled samples
            combined_batch = balanced_batch + random_unlabeled

            # Ensure the final batch size is correct
            if len(combined_batch) == self.batch_size:
                num_batches_generated += 1
                yield combined_batch

    def __len__(self):
        max_class_size = max(len(indices) for indices in self.label_to_indices.values())
        return (max_class_size * self.num_labels) // self.small_batch_size
