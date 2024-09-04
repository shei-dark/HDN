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


def custom_collate_fn(batch):
    patches, labels = zip(*batch)
    patches = torch.stack(patches)
    labels = torch.tensor(labels)
    return patches, labels


class CustomDataset(Dataset):

    def __init__(self, images, labels, patch_size=64, mask_size=4):
        self.patch_size = patch_size
        self.mask_size = mask_size
        self.all_patches = []
        self.name = []
        self.patches_by_label = self._extract_valid_patches(images, labels)
        

    def __len__(self):

        return len(self.all_patches)

    def _extract_valid_patches(self, images, labels):

        patches_by_label = {}
        
        keys = list(images.keys())
        for key in keys:
            slice = 0
            for img, lbl in tqdm(zip(images[key], labels[key]), 'Extracting patches from ' + key):
                height, width = img.shape
                for i in range(0, height // self.patch_size):
                    for j in range(0, width // self.patch_size):
                        x = j * self.patch_size
                        y = i * self.patch_size
                        patch = img[y : y + self.patch_size, x : x + self.patch_size]
                        patch_label = lbl[y : y + self.patch_size, x : x + self.patch_size]
                        blind_spot_area = patch_label[
                            self.patch_size // 2 - self.mask_size // 2 : self.patch_size // 2 + self.mask_size // 2 + 1,
                            self.patch_size // 2 - self.mask_size // 2 : self.patch_size // 2 + self.mask_size // 2 + 1,
                        ]
                        unique_labels = np.unique(blind_spot_area)
                            # without background
                        # if len(unique_labels) == 1 and unique_labels[0] != 0:
                            # with background
                        # -1 is for outside of the cell
                        if len(unique_labels) == 1 and unique_labels[0] != -1:
                            center_label = unique_labels[0]
                            if center_label not in patches_by_label:
                                patches_by_label[center_label] = []
                            self.all_patches.append((torch.tensor(patch).unsqueeze(0), torch.tensor(center_label), torch.tensor(patch_label).unsqueeze(0)))
                            self.name.append(key+"_"+str(slice)+"_"+str(y)+"_"+str(x))
                            patches_by_label[center_label].append(len(self.all_patches) - 1)
                slice += 1
        return patches_by_label

    def __getitem__(self, idx):
        if isinstance(idx, list):
            patches = [self.all_patches[i] for i in idx]
            names = [self.name[i] for i in idx]
            patches, clss, labels = zip(*patches)
            return torch.stack(patches), names, torch.stack(labels)
        else:
            patch, cls, label = self.all_patches[idx]
            return patch, cls, label


class CropAugDataset(CustomDataset):
    
    def __init__(self, images, labels, patch_size=64, mask_size=4):
        self.patch_size = patch_size
        self.mask_size = mask_size
        self.all_patches = []
        self.patches_by_label = self._extract_valid_patches(images, labels)

    def __len__(self):

        return len(self.all_patches)
    
    def _extract_valid_patches(self, images, labels):

        patches_by_label = {}

        keys = list(images.keys())
        for key in keys:
            for img, lbl in tqdm(zip(images[key], labels[key]), 'Extracting patches from ' + key):
                found = False
                height, width = img.shape
                covered = np.zeros((height, width), dtype=bool)
                while not found:
                    y = random.randrange(0, height)
                    x = random.randrange(0, width)
                    if y + self.patch_size*2 <= height and x + self.patch_size*2 <= width:
                        patch = img[y : y + self.patch_size*2, x : x + self.patch_size*2]
                        patch_label = lbl[y : y + self.patch_size*2, x : x + self.patch_size*2]
                        blind_spot_area = patch_label[
                            self.patch_size - self.mask_size : self.patch_size + self.mask_size + 1,
                            self.patch_size - self.mask_size : self.patch_size + self.mask_size + 1,
                        ]
                        unique_labels = np.unique(blind_spot_area)
                        # -1 is for outside of the cell
                        if len(unique_labels) == 1 and unique_labels[0] != -1:
                            center_label = unique_labels[0]
                            if center_label not in patches_by_label:
                                patches_by_label[center_label] = []
                            patch, patch_label = self._downsample(patch, patch_label)
                            self.all_patches.append((torch.tensor(patch), torch.tensor(center_label), torch.tensor(patch_label)))
                            patches_by_label[center_label].append(len(self.all_patches) - 1)
                            covered[y + self.patch_size - self.mask_size : y + self.patch_size + self.mask_size + 1,\
                                    x + self.patch_size - self.mask_size : x + self.patch_size + self.mask_size + 1,] = True

                
                for y in range(0, height, self.patch_size):
                    for x in range(0, width, self.patch_size):
                        if y + self.patch_size <= height and x + self.patch_size <= width:
                            if not covered[y + self.patch_size // 2 - self.mask_size // 2 :\
                                        y + self.patch_size // 2 + self.mask_size // 2 + 1,\
                                        x + self.patch_size // 2 - self.mask_size // 2 :\
                                        x + self.patch_size // 2 + self.mask_size // 2 + 1].any():
                                patch = img[y : y + self.patch_size, x : x + self.patch_size]
                                patch_label = lbl[y : y + self.patch_size, x : x + self.patch_size]
                                blind_spot_area = patch_label[
                                    self.patch_size // 2 - self.mask_size // 2 : self.patch_size // 2 + self.mask_size // 2 + 1,
                                    self.patch_size // 2 - self.mask_size // 2 : self.patch_size // 2 + self.mask_size // 2 + 1,
                                ]
                                unique_labels = np.unique(blind_spot_area)
                                # -1 is for outside of the cell
                                if len(unique_labels) == 1 and unique_labels[0] != -1:
                                    center_label = unique_labels[0]
                                    if center_label not in patches_by_label:
                                        patches_by_label[center_label] = []
                                    self.all_patches.append((torch.tensor(patch).unsqueeze(0), torch.tensor(center_label), torch.tensor(patch_label).unsqueeze(0)))
                                    patches_by_label[center_label].append(len(self.all_patches) - 1)
        return patches_by_label


    def _downsample(self, img, lbl):
        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
        img_downsampled = F.interpolate(img, size=(self.patch_size, self.patch_size),\
                                         mode='bilinear', align_corners=False)
        lbl = torch.from_numpy(lbl).unsqueeze(0).unsqueeze(0)
        lbl_downsampled = F.interpolate(lbl.float(), size=(self.patch_size, self.patch_size),\
                                         mode='nearest')
        return img_downsampled.squeeze(0), lbl_downsampled.squeeze(0)
    
    def __getitem__(self, idx):
        if isinstance(idx, list):
            patches = [self.all_patches[i] for i in idx]
            patches, clss, labels = zip(*patches)
            return torch.stack(patches), torch.tensor(clss), torch.stack(labels)
        else:
            patch, cls, label = self.all_patches[idx]
            return patch, cls, label
    
class CustomTestDataset(Dataset):

    def __init__(self, images, labels, patch_size=64, mask_size=4):
        self.patch_size = patch_size
        self.mask_size = mask_size
        self.all_patches = []
        self.patches_by_label = self._extract_valid_patches(images, labels)
        

    def __len__(self):

        return len(self.all_patches)

    def _extract_valid_patches(self, images, labels):

        patches_by_label = {}
        
        
        for idx in tqdm(range(len(images)), 'Extracting patches from high_c4'):
            img = images[idx]
            lbl = labels[idx]
            height, width = img.shape
            for i in range(0, height // self.patch_size):
                for j in range(0, width // self.patch_size):
                    x = j * self.patch_size
                    y = i * self.patch_size
                    patch = img[y : y + self.patch_size, x : x + self.patch_size]
                    patch_label = lbl[y : y + self.patch_size, x : x + self.patch_size]
                    blind_spot_area = patch_label[
                        self.patch_size // 2 - self.mask_size // 2 : self.patch_size // 2 + self.mask_size // 2 + 1,
                        self.patch_size // 2 - self.mask_size // 2 : self.patch_size // 2 + self.mask_size // 2 + 1,
                    ]
                    unique_labels = np.unique(blind_spot_area)
                        # without background
                    # if len(unique_labels) == 1 and unique_labels[0] != 0:
                        # with background
                    # -1 is for outside of the cell
                    if len(unique_labels) == 1 and unique_labels[0] != -1:
                            center_label = unique_labels[0]
                            if center_label not in patches_by_label:
                                patches_by_label[center_label] = []
                            if len(patches_by_label[center_label]) < 1000:
                                self.all_patches.append((torch.tensor(patch).unsqueeze(0), torch.tensor(center_label), torch.tensor(patch_label).unsqueeze(0)))
                                patches_by_label[center_label].append(len(self.all_patches) - 1)
        return patches_by_label

    def __getitem__(self, idx):
        if isinstance(idx, list):
            patches = [self.all_patches[i] for i in idx]
            patches, clss, labels = zip(*patches)
            return torch.stack(patches), torch.tensor(clss), torch.stack(labels)
        else:
            patch, cls, label = self.all_patches[idx]
            return patch, cls, label
        

class BalancedBatchSampler(Sampler):

    """A custom sampler that generates balanced batches from a dataset by ensuring each batch
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
       pass"""
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


class UnbalancedBatchSampler(Sampler):

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

        # dictionary mapping labels to indices
        self.label_to_indices = dataset.patches_by_label
    
        # Determine number of labels
        self.num_labels = len(self.label_to_indices)
        self.samples_per_label = [self.batch_size // 3, self.batch_size // 6, self.batch_size // 3, self.batch_size // 6]
        self.remaining_samples = self.batch_size - np.sum(self.samples_per_label)
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
