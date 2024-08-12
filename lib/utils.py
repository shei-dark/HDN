import torch
import numpy as np
import time
from torch import nn
from tqdm import tqdm
from glob import glob
from sklearn.feature_extraction import image
from matplotlib import pyplot as plt
from skimage.measure import regionprops
import torch.nn.functional as F
from torch.distributions import kl_divergence
from torch.distributions.normal import Normal



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
    
class WeightScheduler:
    def __init__(self, alpha_start, beta_start, alpha_end, beta_end, num_steps):
        self.alpha_start = alpha_start
        self.beta_start = beta_start
        self.alpha_end = alpha_end
        self.beta_end = beta_end
        self.num_steps = num_steps
        self.step_count = 0

    def get_weights(self):
        alpha = self.alpha_start + (self.alpha_end - self.alpha_start) * (self.step_count / self.num_steps)
        beta = self.beta_start + (self.beta_end - self.beta_start) * (self.step_count / self.num_steps)
        return alpha, beta

    def step(self):
        if self.step_count < self.num_steps:
            self.step_count += 1

def update_loss_weights(w_kl, w_cl, kl_loss, cl_loss, inpainting_loss, factor=0.1):
    # Adjust w_kl
    if kl_loss > inpainting_loss:
        w_kl *= (1 - factor)
    else:
        w_kl *= (1 + factor)

    # Adjust w_cl
    if cl_loss > inpainting_loss:
        w_cl *= (1 - factor)
    else:
        w_cl *= (1 + factor)

    # Ensure weights are within a reasonable range
    w_kl = max(0.001, min(1000, w_kl))
    w_cl = max(0.001, min(1000, w_cl))

    return w_kl, w_cl

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

def augment_data(X_train):
    """Augment data by 8-fold with 90 degree rotations and flips. 
    Parameters
    ----------
    X_train: numpy array
        Array of training images.
    """
    X_ = X_train.copy()

    X_train_aug = np.concatenate((X_train, np.rot90(X_, 1, (1, 2))))
    X_train_aug = np.concatenate((X_train_aug, np.rot90(X_, 2, (1, 2))))
    X_train_aug = np.concatenate((X_train_aug, np.rot90(X_, 3, (1, 2))))
    X_train_aug = np.concatenate((X_train_aug, np.flip(X_train_aug, axis=1)))

    print('Raw image size after augmentation', X_train_aug.shape)
    return X_train_aug

def extract_patches(x,y,patch_size,num_patches):
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
    patches = np.zeros(shape=(x.shape[0]*num_patches,patch_size,patch_size))
    labels = np.zeros(shape=(x.shape[0]*num_patches,patch_size,patch_size))
    # for i in tqdm(range(x.shape[0])):
    #     patches[i*num_patches:(i+1)*num_patches] = image.extract_patches_2d(x[i],(patch_size,patch_size), max_patches=num_patches,
    #                                                                        random_state=i)
    #     labels[i*num_patches:(i+1)*num_patches] = image.extract_patches_2d(y[i],(patch_size,patch_size), max_patches=num_patches,random_state=i)    
    # return patches, labels

    # patches = []
    # labels = []
    count = 0
    for i in tqdm(range(x.shape[0])):
        while count < num_patches:
            props = regionprops(y[i])
            for prop in props:
                random_number = np.random.randint(len(prop['coords']))
                x_c = prop['coords'][random_number][0]
                y_c = prop['coords'][random_number][1]
                while x_c < 30 or y_c < 30 or x_c > (len(x[i]) - 34) or y_c > (len(x[i][0]) - 34):
                    random_number = np.random.randint(len(prop['coords']))
                    x_c = prop['coords'][random_number][0]
                    y_c = prop['coords'][random_number][1]
                negh = 0
                for j in range(x_c,x_c+4):
                    for k in range(y_c,y_c+4):
                        if [j,k] in prop['coords']:
                            negh += 1
                if negh == 16:
                    x_coord = x_c - 30
                    y_coord = y_c - 30
                    patches[i*(count+1):(i+1)*(count+1)] = x[i,x_coord:x_coord+patch_size, y_coord:y_coord+patch_size]
                    labels[i*(count+1):(i+1)*(count+1)] = y[i,x_coord:x_coord+patch_size, y_coord:y_coord+patch_size]
                    count += 1
        count = 0
    return patches, labels


def crop_img_tensor(x, size) -> torch.Tensor:
    """Crops a tensor.
    Crops a tensor of shape (batch, channels, h, w) to new height and width
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
    Pads or crops a tensor of shape (batch, channels, h, w) to new height
    and width given by a tuple.
    Args:
        x (torch.Tensor): Input image
        size (list or tuple): Desired size (height, width)
        mode (str): Mode, either 'pad' or 'crop'
    Returns:
        The padded or cropped tensor
    """

    assert x.dim() == 4 and len(size) == 2
    size = tuple(size)
    x_size = x.size()[2:4]
    if mode == 'pad':
        cond = x_size[0] > size[0] or x_size[1] > size[1]
    elif mode == 'crop':
        cond = x_size[0] < size[0] or x_size[1] < size[1]
    else:
        raise ValueError("invalid mode '{}'".format(mode))
    if cond:
        raise ValueError('trying to {} from size {} to size {}'.format(
            mode, x_size, size))
    dr, dc = (abs(x_size[0] - size[0]), abs(x_size[1] - size[1]))
    dr1, dr2 = dr // 2, dr - (dr // 2)
    dc1, dc2 = dc // 2, dc - (dc // 2)
    if mode == 'pad':
        return nn.functional.pad(x, [dc1, dc2, dr1, dr2, 0, 0, 0, 0])
    elif mode == 'crop':
        return x[:, :, dr1:x_size[0] - dr2, dc1:x_size[1] - dc2]
    

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

def compute_cl_loss(mus, logvars, labels, cl_mode, margin):
    """
    mus: (hierarchy levels, batch_size, C, H, W) list
    logvars: (hierarchy levels, batch_size, C, H, W) list
    labels: (batch_size, H, W) -> 64/128, 64x64 tensor
    """
    # --------------
    positive_loss, negative_losses = class_wise_contrastive_loss(mus, labels, num_classes=4)
    normalized_negative_losses = normalize_losses(negative_losses)
    alphas = compute_alphas(normalized_negative_losses)
    cl_loss = compute_total_contrastive_loss(positive_loss, negative_losses, alphas)

    return cl_loss, negative_losses
    # --------------

    # return contrastive_loss(mus, labels, margin)
    # return contrastive_kl_loss(mus, logvars, labels)

def contrastive_kl_loss(mus, logvars, labels, margin=50.0):
    torch.cuda.empty_cache()
    batch_size = len(mus[0])
    n_channel = mus[0].shape[1]
    dist = torch.zeros(batch_size,batch_size).to(device=mus[0].device)
    # Compute pairwise distances
    for i in range(dist.shape[0]):
        p_mu = torch.mean(mus[0][i]) + torch.mean(mus[1][i]) + torch.mean(mus[2][i])
        p_lv = torch.mean(logvars[0][i]) + torch.mean(logvars[1][i]) + torch.mean(logvars[2][i])
        p = Normal(p_mu, (p_lv / 2).exp())
        for j in range(i+1,dist.shape[1]):
            q_mu = torch.mean(mus[0][j]) + torch.mean(mus[1][j]) + torch.mean(mus[2][j])
            q_lv = torch.mean(logvars[0][j]) + torch.mean(logvars[1][j]) + torch.mean(logvars[2][j])
            q = Normal(q_mu, (q_lv / 2).exp())
            dist[i,j] = kl_divergence(p,q)
                                  
            # dist[i,j] = torch.sum(torch.mean(kl_divergence(Normal(mus[0][i], (logvars[0][i] / 2).exp()), Normal(mus[0][j], (logvars[0][j] / 2).exp())).reshape(n_channel,-1),-1)+ \
            #             torch.mean(kl_divergence(Normal(mus[1][i], (logvars[1][i] / 2).exp()), Normal(mus[1][j], (logvars[1][j] / 2).exp())).reshape(n_channel,-1),-1)+ \
            #             torch.mean(kl_divergence(Normal(mus[2][i], (logvars[2][i] / 2).exp()), Normal(mus[2][j], (logvars[2][j] / 2).exp())).reshape(n_channel,-1),-1))
    boolean_matrix = (labels == labels.T).to(device=mus[0].device)
    
    # Positive pairs: minimize distance
    positive_loss = torch.sum(boolean_matrix * dist)
    # Negative pairs: maximize distance (with margin)
    negative_loss = torch.sum(~boolean_matrix * F.relu(margin - dist))
    
    # Normalize by number of pairs
    num_positive_pairs = torch.sum(boolean_matrix)-batch_size
    num_negative_pairs = torch.sum(~boolean_matrix)
    if num_positive_pairs == 0:
        print('No positive pairs')
    if num_negative_pairs == 0:
        print('No negative pairs')

    if num_positive_pairs == 0:
        positive_loss = 0
    else:
        positive_loss /= (num_positive_pairs/2)
    if num_negative_pairs == 0:
        negative_loss = 0
    else:
        negative_loss /= (num_negative_pairs/2)
    
    # Total contrastive loss
    loss = positive_loss + negative_loss
    return loss

def contrastive_loss(z, labels, margin=50.0):
    # Compute pairwise distances
    batch_size = len(z[0])
    z = [z[i].reshape(batch_size, -1) for i in range(len(z))]
    z = torch.cat(z,dim=-1).unsqueeze(0)
    dist = torch.cdist(z, z, p=2).squeeze(0)
    
    boolean_matrix = (labels == labels.T).to(device=z.device)
    
    # Positive pairs: minimize distance
    positive_loss = torch.sum(boolean_matrix * dist)
    # Negative pairs: maximize distance (with margin)
    negative_loss = torch.sum(~boolean_matrix * F.relu(margin - dist))
    
    # Normalize by number of pairs
    num_positive_pairs = torch.sum(boolean_matrix)-batch_size
    num_negative_pairs = torch.sum(~boolean_matrix)
    if num_positive_pairs == 0:
        print('No positive pairs')
    if num_negative_pairs == 0:
        print('No negative pairs')

    if num_positive_pairs == 0:
        positive_loss = 0
    else:
        positive_loss /= num_positive_pairs
    if num_negative_pairs == 0:
        negative_loss = 0
    else:
        negative_loss /= num_negative_pairs
    
    # Total contrastive loss
    loss = positive_loss + negative_loss
    return loss

def class_wise_contrastive_loss(z, labels, num_classes=4):
    # Compute pairwise distances
    batch_size = len(z[0])
    z = [z[i].reshape(batch_size, -1) for i in range(len(z))]
    z = torch.cat(z, dim=-1).unsqueeze(0)
    dist = torch.cdist(z, z, p=2).squeeze(0)
    
    # Compute positive pairs loss
    boolean_matrix = (labels == labels.T).to(device=z.device)
    positive_loss = torch.sum(boolean_matrix * dist)
    
    # Normalize positive loss
    num_positive_pairs = torch.sum(boolean_matrix) - batch_size
    if num_positive_pairs == 0:
        positive_loss = 0
    else:
        positive_loss /= num_positive_pairs
    
    # Compute negative pairs loss per class pair
    negative_losses = {}
    for i in range(num_classes-1):
        for j in range(i+1, num_classes):
            mask_i = (labels == i).unsqueeze(1)
            mask_j = (labels == j).unsqueeze(1)
            mask_ij = (mask_i & mask_j.T).squeeze(1)
            
            neg_boolean_matrix = mask_ij.to(device=z.device)
            negative_loss = torch.sum(neg_boolean_matrix * dist)
            
            num_negative_pairs = torch.sum(neg_boolean_matrix)
            if num_negative_pairs == 0:
                negative_loss = 0
            else:
                negative_loss /= num_negative_pairs
            
            negative_losses[f'{i}{j}'] = negative_loss
    
    return positive_loss, negative_losses

def normalize_losses(negative_losses):
    losses = torch.tensor(list(negative_losses.values()))
    normalized_losses = (losses - losses.min()) / (losses.max() - losses.min())
    normalized_dict = {key: normalized_losses[i].item() for i, key in enumerate(negative_losses.keys())}
    return normalized_dict

def compute_alphas(normalized_losses):
    alphas = {key: 1.0 - value for key, value in normalized_losses.items()}
    for key, value in alphas.items():
        if value == 0:
            alphas[key] = 0.0001
        if value == 1:
            alphas[key] = 0.9999
    return alphas

def compute_total_contrastive_loss(positive_loss, negative_losses, alphas):
    weighted_negative_loss = 0
    target_negative_loss = 20.0
    for pair, loss in negative_losses.items():
        weighted_negative_loss += alphas.get(pair, 1.0) * loss

    # Penalize based on how far the weighted_negative_loss is from the target
    negative_loss_penalty = torch.abs(weighted_negative_loss - target_negative_loss)
    
    
    # Total contrastive loss: minimize positive loss and the deviation of weighted negative loss from target
    total_contrastive_loss = positive_loss + negative_loss_penalty
    return total_contrastive_loss