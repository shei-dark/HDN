import torch
import numpy as np
import time
from torch import nn
from tqdm import tqdm
from glob import glob
from sklearn.feature_extraction import image
from skimage.util import view_as_windows
from matplotlib import pyplot as plt


class Interpolate(nn.Module):
    """Wrapper for torch.nn.functional.interpolate."""

    def __init__(self,
                 size=None,
                 scale=None,
                 mode='bilinear',
                 align_corners=False):
        super().__init__()
        assert (size is None) == (scale is not None)
        self.size = size
        self.scale = scale
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        out = F.interpolate(x,
                            size=self.size,
                            scale_factor=self.scale,
                            mode=self.mode,
                            align_corners=self.align_corners)
        return out
    

class CropImage(nn.Module):
    """Crops image to given size.
    Args:
        size
    """

    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        return crop_img_tensor(x, self.size)
    


def normalize(img, mean, std):
    """Normalize an array of images with mean and standard deviation. 
        Parameters
        ----------
        img: array
            An array of images.
        mean: float
            Mean of img array.
        std: float
            Standard deviation of img array.
        """
    return (img - mean)/std


def denormalize(img, mean, std):
    """Denormalize an array of images with mean and standard deviation. 
    Parameters
    ----------
    img: array
        An array of images.
    mean: float
        Mean of img array.
    std: float
        Standard deviation of img array.
    """
    return (img * std) + mean


def convertToFloat32(train_images,val_images):
    """Converts the data to float 32 bit type. 
    Parameters
    ----------
    train_images: array
        Training data.
    val_images: array
        Validation data.
        """
    x_train = train_images.astype('float32')
    x_val = val_images.astype('float32')
    return x_train, x_val


def getMeanStdData(train_images,val_images):
    """Compute mean and standrad deviation of data. 
    Parameters
    ----------
    train_images: array
        Training data.
    val_images: array
        Validation data.
    """
    x_train_ = train_images.astype('float32')
    x_val_ = val_images.astype('float32')
    data = np.concatenate((x_train_,x_val_), axis=0)
    mean, std = np.mean(data), np.std(data)
    return mean, std


def convertNumpyToTensor(numpy_array):
    """Convert numpy array to PyTorch tensor. 
    Parameters
    ----------
    numpy_array: numpy array
        Numpy array.
    """
    return torch.from_numpy(numpy_array)


def augment_data(patches):
    if len(patches.shape[1:]) == 2:
        augmented = np.concatenate((patches,
                                    np.rot90(patches, k=1, axes=(1, 2)),
                                    np.rot90(patches, k=2, axes=(1, 2)),
                                    np.rot90(patches, k=3, axes=(1, 2))))
    elif len(patches.shape[1:]) == 3:
        augmented = np.concatenate((patches,
                                    np.rot90(patches, k=1, axes=(2, 3)),
                                    np.rot90(patches, k=2, axes=(2, 3)),
                                    np.rot90(patches, k=3, axes=(2, 3))))

    augmented = np.concatenate((augmented, np.flip(augmented, axis=-2)))
    return augmented


def extract_patches(x, patch_size, num_patches):
    """Deterministically extract patches from array of images. 
    Parameters
    ----------
    x: numpy array
        Array of images.
    patch_size: int
        Size of patches to be extracted from each image.
    num_patches: int
        Number of patches to be extracted from each image.    
    """
    patches = np.zeros(shape=(x.shape[0]*num_patches, patch_size, patch_size))
    
    for i in tqdm(range(x.shape[0])):
        patches[i*num_patches:(i+1)*num_patches] = image.extract_patches_2d(image=x[i],
                                                                            patch_size=(patch_size, patch_size),
                                                                            max_patches=num_patches, 
                                                                            random_state=i)    
    
    return patches




def crop_img_tensor(x, size) -> torch.Tensor:
    """Crops a tensor.
    Crops a tensor of shape (batch, channels, h, w) or (batch, channels, d, h, w) to new height and width
    given by a tuple.
    Args:
        x (torch.Tensor): Input image
        size (list or tuple): Desired size (height, width)
    Returns:
        The cropped tensor
    """
    return _pad_crop_img(x, size, 'crop')


def _pad_crop_img(x, size, mode) -> torch.Tensor:
    """ Pads or crops a tensor.
    Pads or crops a tensor of shape (batch, channels, h, w) or (batch, channels, d, h, w) to new height
    and width given by a tuple.
    Args:
        x (torch.Tensor): Input image
        size (list or tuple): Desired size (depth: opt, height, width)
        mode (str): Mode, either 'pad' or 'crop'
    Returns:
        The padded or cropped tensor
    """
    assert x.dim() in [4, 5], 'Invalid input array dimension'
    assert len(size) in [2, 3], 'Invalid input depth dimension'
    # assert
    size = tuple(size)
    x_size = x.size()[2:]

    if mode == 'pad':
        cond = any(x_size) > any(size)
    elif mode == 'crop':
        cond = any(x_size) < any(size)
    else:
        raise ValueError(f'invalid mode {mode}')

    if cond:
        raise ValueError(f'trying to {mode} from size {x_size} to size {size}')

    padding = []
    for d in reversed(range(len(x_size))):
        pad_val = abs(x_size[d] - size[d])
        padding.append(pad_val // 2)
        padding.append(pad_val - (pad_val // 2))

    if mode == 'pad':
        return nn.functional.pad(x, padding)
    elif mode == 'crop':
        if len(x_size) == 2:
            return x[:, :, padding[2]:x_size[0] - padding[3], padding[0]:x_size[1] - padding[1]]
        elif len(x_size) == 3:
            return x[:, :, padding[4]:x_size[0] - padding[5], padding[2]:x_size[1] - padding[3],
                   padding[0]:x_size[2] - padding[1]]
    

def free_bits_kl(kl,
                 free_bits,
                 batch_average = False,
                 eps = 1e-6) -> torch.Tensor:
    """Computes free-bits version of KL divergence.
    Takes in the KL with shape (batch size, layers), returns the KL with
    free bits (for optimization) with shape (layers,), which is the average
    free-bits KL per layer in the current batch.
    If batch_average is False (default), the free bits are per layer and
    per batch element. Otherwise, the free bits are still per layer, but
    are assigned on average to the whole batch. In both cases, the batch
    average is returned, so it's simply a matter of doing mean(clamp(KL))
    or clamp(mean(KL)).
    Args:
        kl (torch.Tensor)
        free_bits (float)
        batch_average (bool, optional))
        eps (float, optional)
    Returns:
        The KL with free bits
    """

    assert kl.dim() == 2
    if free_bits < eps:
        return kl.mean(0)
    if batch_average:
        return kl.mean(0).clamp(min=free_bits)
    return kl.clamp(min=free_bits).mean(0)


def pad_img_tensor(x, size) -> torch.Tensor:
    """Pads a tensor.
    Pads a tensor of shape (batch, channels, h, w) to new height and width
    given by a tuple.
    Args:
        x (torch.Tensor): Input image
        size (list or tuple): Desired size (height, width)
    Returns:
        The padded tensor
    """

    return _pad_crop_img(x, size, 'pad')


def plotProbabilityDistribution(signalBinIndex, histogram, gaussianMixtureNoiseModel, min_signal, max_signal, n_bin, device):
    """Plots probability distribution P(x|s) for a certain ground truth signal. 
       Predictions from both Histogram and GMM-based Noise models are displayed for comparison.
        Parameters
        ----------
        signalBinIndex: int
            index of signal bin. Values go from 0 to number of bins (`n_bin`).
        histogram: numpy array
            A square numpy array of size `nbin` times `n_bin`.
        gaussianMixtureNoiseModel: GaussianMixtureNoiseModel
            Object containing trained parameters.
        min_signal: float
            Lowest pixel intensity present in the actual sample which needs to be denoised.
        max_signal: float
            Highest pixel intensity present in the actual sample which needs to be denoised.
        n_bin: int
            Number of Bins.
        device: GPU device
        """
    histBinSize=(max_signal-min_signal)/n_bin
    querySignal_numpy= (signalBinIndex/float(n_bin)*(max_signal-min_signal)+min_signal)
    querySignal_numpy +=histBinSize/2
    querySignal_torch = torch.from_numpy(np.array(querySignal_numpy)).float().to(device)
    
    queryObservations_numpy=np.arange(min_signal, max_signal, histBinSize)
    queryObservations_numpy+=histBinSize/2
    queryObservations = torch.from_numpy(queryObservations_numpy).float().to(device)
    pTorch=gaussianMixtureNoiseModel.likelihood(queryObservations, querySignal_torch)
    pNumpy=pTorch.cpu().detach().numpy()
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.xlabel('Observation Bin')
    plt.ylabel('Signal Bin')
    plt.imshow(histogram**0.25, cmap='gray')
    plt.axhline(y=signalBinIndex+0.5, linewidth=5, color='blue', alpha=0.5)
    
    plt.subplot(1, 2, 2)
    plt.plot(queryObservations_numpy, histogram[signalBinIndex, :]/histBinSize, label='GT Hist: bin ='+str(signalBinIndex), color='blue', linewidth=2)
    plt.plot(queryObservations_numpy, pNumpy, label='GMM : '+' signal = '+str(np.round(querySignal_numpy,2)), color='red',linewidth=2)
    plt.xlabel('Observations (x) for signal s = ' + str(querySignal_numpy))
    plt.ylabel('Probability Density')
    plt.title("Probability Distribution P(x|s) at signal =" + str(querySignal_numpy))
    plt.legend()
    

def PSNR(gt, img, psnrRange):
    '''
    Compute PSNR.
    Parameters
    ----------
    gt: array
        Ground truth image.
    img: array
        Predicted image.
    psnrRange: float
        Range PSNR
    '''
    mse = np.mean(np.square(gt - img))
    return 20 * np.log10(psnrRange) - 10 * np.log10(mse)


def get_normalized_tensor(img,model,device):
    '''
    Normalizes tensor with mean and std.
    Parameters
    ----------
    img: array
        Image.
    model: Hierarchical DivNoising model
    device: GPU device.
    '''
    test_images = torch.from_numpy(img).to(device)
    data_mean = model.data_mean
    data_std = model.data_std
    test_images = (test_images-data_mean)/data_std
    return test_images

def compute_cl_loss(mus, labels, margin=50, lambda_contrastive=0.5, labeled_ratio=1):

    output = {}
    pos_pair_loss, neg_pair_loss_terms = pos_neg_loss(mus, labels, margin, labeled_ratio)
    neg_thetas = get_thetas(neg_pair_loss_terms)
    weighted_neg = compute_weighted_neg(neg_pair_loss_terms, neg_thetas)
    contrastive_loss = lambda_contrastive * pos_pair_loss + (1-lambda_contrastive) * weighted_neg

    output = {
        'cl_loss': contrastive_loss,
        'pos_pair_loss': pos_pair_loss,
        'neg_pair_loss': weighted_neg,
        'neg_pair_terms': neg_pair_loss_terms,
        'thetas': neg_thetas,
    }
    return output

def pos_neg_loss(mus, labels, margin=50, labeled_ratio=1):

    num_classes = torch.unique(labels).size(0)
    batch_size = mus.size(0)
    small_batch_size = int(batch_size * labeled_ratio)
    mus = [mus[i].view(batch_size, -1) for i in range(len(mus))]
    mus = torch.cat(mus, dim=-1).undqueeze(0)
    mus = mus[:, :small_batch_size,...]
    dist = torch.cdist(mus, mus, p=2).squeeze(0)

    labels = labels[:, :small_batch_size]
    boolean_matrix = (labels == labels.T).to(device=mus.device)
    pos_pair_loss = torch.sum(boolean_matrix * dist)

    num_pos_pairs = torch.sum(boolean_matrix) - small_batch_size
    if num_pos_pairs == 0:
        pos_pair_loss = 0
    else:
        pos_pair_loss /= num_pos_pairs

    neg_pair_loss_terms = {}
    for i in range(num_classes-1):
        for j in range(i+1, num_classes):
            mask_i = (labels == i).unsqueeze(1)
            mask_j = (labels == j).unsqueeze(1)
            mask_ij = (mask_i & mask_j.T).squeeze(1)

            neg_bool_matrix = mask_ij.to(device=mus.device)
            neg_loss = torch.sum(neg_bool_matrix * F.relu(margin - dist))

            num_neg_pairs = torch.sum(neg_bool_matrix)
            if num_neg_pairs == 0:
                neg_loss = 0
            else:
                neg_loss /= num_neg_pairs

            neg_pair_loss_terms[f'{i}{j}'] = neg_loss

    return pos_pair_loss, neg_pair_loss_terms


def get_thetas(neg_pair_loss_terms):

    losses = torch.tensor(list(neg_pair_loss_terms.values()))
    normalized_losses = F.softmax(losses, dim=0)
    thetas = {key: normalized_losses[i].item() for i, key in enumerate(neg_pair_loss_terms.keys())}
    return thetas


def compute_weighted_neg(neg_pair_loss_terms, neg_thetas):

    weighted_neg = 0

    for pair, loss in neg_pair_loss_terms.items():
        weighted_neg += neg_thetas.get(pair, 1.0) * loss

    return weighted_neg