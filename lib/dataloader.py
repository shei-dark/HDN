import queue
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
import random
from PIL import Image
from glob import glob
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """
    A custom dataset class for handling image patches and labels.

    Args:
        images (list): List of input images.
        labels (list): List of corresponding labels.
        patch_size (int, optional): Size of the image patches. Defaults to 64.
        mask_size (int, optional): Size of the blind spot mask. Defaults to 4.
    """

    def __init__(self, images, labels, patch_size=64, mask_size=4):
        self.patch_size = patch_size
        self.mask_size = mask_size
        self.patches_by_label = self._extract_valid_patches(images, labels)

    def _extract_valid_patches(self, images, labels):
        """
        Extracts valid patches from the input images and labels.

        Args:
            images (list): List of input images.
            labels (list): List of corresponding labels.

        Returns:
            dict: Dictionary containing patches grouped by label.
        """
        patches_by_label = {}
        
        for img, lbl in zip(images, labels):
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
                    if len(unique_labels) == 1 and unique_labels[0] != 0:
                            center_label = unique_labels[0]
                            if center_label not in patches_by_label:
                                patches_by_label[center_label] = []
                            patches_by_label[center_label].append(
                                torch.tensor(patch, dtype=torch.float32).unsqueeze(0)
                            )
        return patches_by_label


    def __getitem__(self, lbl_id, num_patches=1):
        """
        Retrieves a specified number of patches for a given label.

        Args:
            lbl_id (int): Label ID.
            num_patches (int, optional): Number of patches to retrieve. Defaults to 1.

        Returns:
            list or torch.Tensor: List of selected patches or a single patch as a tensor.
        """
        patches = self.patches_by_label[lbl_id]
        try:
            selected_patches = random.sample(patches, num_patches)
        except ValueError as e:
            print(f"Number of patches requested ({num_patches}) exceeds the number of available patches ({len(patches)}).")
            selected_patches = random.choice(patches)
        return selected_patches


# class MultiClassSampler(Sampler):
#     """
#     A custom sampler that generates batches with balanced class distribution.

#     Args:
#         labels (numpy.ndarray): Array of labels for each sample.
#         batch_size (int): The desired batch size.

#     Attributes:
#         labels (numpy.ndarray): Array of labels for each sample.
#         batch_size (int): The desired batch size.
#         label_to_indices (dict): Dictionary mapping each label to the indices of samples with that label.

#     Methods:
#         __iter__(): Generates batches with balanced class distribution.
#         __len__(): Returns the total number of samples.

#     """

#     def __init__(self, labels, batch_size):
#         self.labels = labels
#         self.batch_size = batch_size
#         self.label_to_indices = {}
#         for label in set(labels):
#             self.label_to_indices[label] = np.where(labels == label)[0]

#     def __iter__(self):
#         batches = []
#         for _ in range(len(self) // self.batch_size):
#             batch = []
#             for label in self.label_to_indices:
#                 batch.extend(
#                     np.random.choice(
#                         self.label_to_indices[label],
#                         self.batch_size // len(self.label_to_indices),
#                         replace=False,
#                     )
#                 )
#             np.random.shuffle(batch)
#             batches.append(batch)
#         np.random.shuffle(batches)
#         return iter(batches)

#     def __len__(self):
#         return len(self.labels)


# class MemoryBank:
#     """
#     A class representing a memory bank for storing features and labels.

#     Args:
#         feature_dim (int): The dimension of the feature vectors.
#         memory_size (int): The maximum number of feature vectors that can be stored in the memory bank.

#     Attributes:
#         memory (torch.Tensor): The memory bank for storing feature vectors.
#         memory_labels (torch.Tensor): The labels corresponding to the feature vectors in the memory bank.
#         memory_ptr (int): The current pointer position in the memory bank.
#         memory_size (int): The maximum number of feature vectors that can be stored in the memory bank.
#     """

#     def __init__(self, feature_dim, memory_size):
#         self.memory = torch.randn(memory_size, feature_dim)
#         self.memory_labels = torch.zeros(memory_size, dtype=torch.long)
#         self.memory_ptr = 0
#         self.memory_size = memory_size

#     def update(self, features, labels):
#         """
#         Updates the memory bank with new feature vectors and labels.

#         Args:
#             features (torch.Tensor): The feature vectors to be added to the memory bank.
#             labels (torch.Tensor): The labels corresponding to the feature vectors.

#         Returns:
#             None
#         """
#         batch_size = features.size(0)
#         if self.memory_ptr + batch_size > self.memory_size:
#             overflow = self.memory_ptr + batch_size - self.memory_size
#             self.memory[self.memory_ptr :] = features[: batch_size - overflow]
#             self.memory_labels[self.memory_ptr :] = labels[: batch_size - overflow]
#             self.memory_ptr = 0
#             self.memory[:overflow] = features[batch_size - overflow :]
#             self.memory_labels[:overflow] = labels[batch_size - overflow]
#         else:
#             self.memory[self.memory_ptr : self.memory_ptr + batch_size] = features
#             self.memory_labels[self.memory_ptr : self.memory_ptr + batch_size] = labels
#             self.memory_ptr += batch_size

#     def sample_positives(self, features, labels, num_positives):
#         """
#         Samples hard positive feature vectors from the memory bank.

#         Args:
#             features (torch.Tensor): The input feature vectors.
#             labels (torch.Tensor): The labels corresponding to the input feature vectors.
#             num_positives (int): The number of hard positive samples to be retrieved.

#         Returns:
#             torch.Tensor: The hard positive feature vectors.
#         """
#         positives = []
#         for feature, label in zip(features, labels):
#             pos_indices = (
#                 (self.memory_labels == label).nonzero(as_tuple=False).squeeze()
#             )
#             pos_samples = self.memory[pos_indices]
#             distances = torch.norm(pos_samples - feature, dim=1)
#             hard_positives = pos_samples[
#                 distances.topk(num_positives, largest=False).indices
#             ]
#             positives.append(hard_positives)
#         return torch.cat(positives)

#     def sample_negatives(self, features, labels, num_negatives):
#         """
#         Samples hard negative feature vectors from the memory bank.

#         Args:
#             features (torch.Tensor): The input feature vectors.
#             labels (torch.Tensor): The labels corresponding to the input feature vectors.
#             num_negatives (int): The number of hard negative samples to be retrieved.

#         Returns:
#             torch.Tensor: The hard negative feature vectors.
#         """
#         negatives = []
#         for feature, label in zip(features, labels):
#             neg_indices = (
#                 (self.memory_labels != label).nonzero(as_tuple=False).squeeze()
#             )
#             neg_samples = self.memory[neg_indices]
#             distances = torch.norm(neg_samples - feature, dim=1)
#             hard_negatives = neg_samples[
#                 distances.topk(num_negatives, largest=False).indices
#             ]
#             negatives.append(hard_negatives)
#         return torch.cat(negatives)


# class DataLoader:
#     def __init__(self, train_data, test_data):
#         self.train_data = train_data
#         self.test_data = test_data
#         self.train_queue = queue.Queue()
#         self.test_queues = {}

#         # Initialize test queues for each label
#         for label in self.test_data.keys():
#             self.test_queues[label] = queue.Queue()

#         # Populate the train queue
#         self._populate_train_queue()

#     def _populate_train_queue(self):
#         for data in self.train_data:
#             self.train_queue.put(data)

#     def get_train_batch(self, batch_size):
#         batch = []
#         for _ in range(batch_size):
#             try:
#                 data = self.train_queue.get(timeout=1)
#                 batch.append(data)
#             except queue.Empty:
#                 break
#         return batch

#     def get_test_batch(self, label, batch_size):
#         with self.lock:
#             test_queue = self.test_queues[label]
#         batch = []
#         for _ in range(batch_size):
#             try:
#                 data = test_queue.get(timeout=1)
#                 batch.append(data)
#             except queue.Empty:
#                 break
#         return batch

#     def add_test_data(self, label, data):
#         with self.lock:
#             test_queue = self.test_queues[label]
#         test_queue.put(data)

#     def evaluate_test_data(self):
#         for label, test_queue in self.test_queues.items():
#             while not test_queue.empty():
#                 data = test_queue.get()
#                 # Perform evaluation on the test data
#                 # ...

#     def stop(self):
#         self.train_thread.join()
