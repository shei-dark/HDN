import pickle
import torch
from tifffile import imread
import numpy as np


def prep_data(data, patch_size, centre_size, model, device):

    mirrored_img = mirror_image(data, patch_size=patch_size, centre_size=centre_size)
    padded_img = pad_to_divisible_by(mirrored_img, centre_size)
    normalized_img = get_normalized_tensor(padded_img, model, device)
    return normalized_img

def load_data(dir):
    return imread(dir)

def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pickle(file_name, data):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)

def get_normalized_tensor(img,model,device):
    test_images = torch.from_numpy(img.copy()).to(device)
    data_mean = model.data_mean
    data_std = model.data_std
    test_images = (test_images-data_mean)/data_std
    return test_images

def pad_to_divisible_by(image, centre_size):
    """
    Pads the image so that its dimensions are divisible by 4.
    """
    h, w = image.shape[:2]
    
    # Calculate the padding needed for each dimension
    pad_h = (centre_size - h % centre_size) % centre_size
    pad_w = (centre_size - w % centre_size) % centre_size
    
    # Padding in (before, after) format
    padding = ((0, pad_h), (0, pad_w)) + ((0, 0),) * (image.ndim - 2)
    
    # Pad the image
    padded_image = np.pad(image, padding, mode='reflect')
    
    return padded_image

def mirror_image(image, patch_size, centre_size):
    """
    Mirror the borders of the image to handle edge cases.
    """
    pad_size = (patch_size + centre_size)// 2

    mirrored_image = np.pad(image, ((pad_size, pad_size)), mode='reflect')
    return mirrored_image

def get_random_patch(image, label, patch_size):
    """
    Get a random patch from the image.
    """
    h, w = image.shape
    y = np.random.randint(0, h - patch_size)
    x = np.random.randint(0, w - patch_size)
    patch = torch.from_numpy(image[y:y + patch_size, x:x + patch_size])
    lbl = label[y:y + patch_size, x:x + patch_size]
    return patch, lbl, y, x

def get_all_patches(image, label, patch_size, stride):
    """
    Get all patches from the image.
    """
    h, w = image.shape
    patches = []
    lbls = []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = torch.from_numpy(image[y:y + patch_size, x:x + patch_size])
            lbl = label[y:y + patch_size, x:x + patch_size]
            patches.append(patch)
            lbls.append(lbl)
    return patches, lbls