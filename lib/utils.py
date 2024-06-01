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

def euclidean_distance_based(z1, z2, target, size_average=True):
    """Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class,
      and label == 0 otherwise"""
    margin = 1e-2
    eps = 1e-9
    z1 = torch.reshape(z1[0].T,(-1,32))
    z2 = torch.reshape(z2[0].T,(-1,32))
    distances = (z2 - z1).pow(2).sum(1)  # squared distances
    losses = 0.5 * (target * distances +
        (1 + -1 * target) * F.relu(margin - (distances + eps).sqrt()).pow(2))
    return losses.mean() if size_average else losses.sum()

def metric_ed(z1, z2):
    z1 = torch.reshape(z1.T,(-1,32))
    z2 = torch.reshape(z2.T,(-1,32))
    distances = (z2 - z1).pow(2).mean(0)
    return torch.norm(distances)

def metric_cs(z1, z2):
    z1 = torch.reshape(z1[0].T,(-1,32))
    z2 = torch.reshape(z2[0].T,(-1,32))
    return torch.mean(F.cosine_similarity(z1, z2, dim=-1))

def metric_kl(dis1, dis2):
    m1, logv1 = dis1
    m2, logv2 = dis2
    std1 = (logv1 / 2).exp()
    std2 = (logv2 / 2).exp()
    dis1 = Normal(m1, std1)
    dis2 = Normal(m2, std2)
    temp = kl_divergence(dis1, dis2)
    temp = torch.reshape(temp, (-1, 32))
    return torch.mean(temp)

def triplet_loss(anchor, pos, neg):
    # loss = 0
    # for pos_sample in pos:
    #     for neg_sample in neg:
    #         loss += torchtripletloss(anchor, pos_sample[0],neg_sample[0])
    # return loss
    margin = 0
    num_pos = 0
    num_neg = 0
    positive_distance = 0
    negative_distance = 0
    for pos_sample in pos:
        positive_distance = (anchor - pos_sample[0]).pow(2).sum(-1)
        num_pos +=1
    if num_pos != 0:
        positive_distance = positive_distance/num_pos
        positive_distance = positive_distance.mean()
    for neg_sample in neg:
        negative_distance = (anchor - neg_sample[0]).pow(2).sum(-1)
        num_neg += 1
    if num_neg != 0:
        negative_distance = negative_distance/num_neg
        negative_distance = negative_distance.mean()
    return(F.relu(positive_distance - negative_distance + margin))

def compute_cl_loss(mus, logvars, labels, cl_mode):
    """
    mus: (hierarchy levels, batch_size, C, H, W) list
    logvars: (hierarchy levels, batch_size, C, H, W) list
    labels: (batch_size, H, W) -> 64/128, 64x64 tensor
    """
    # ### test new contrastive loss
    # return contrastive_loss(mus, labels)
    # ###
    margin = 1
    labels_list = [0 for _ in range(len(labels))]
    for index, batch_label in enumerate(labels):
        # all pixels in 4x4 centre have same label
        centre = batch_label[30:34,30:34]

        unique_center = torch.unique(centre)
        if 0 not in unique_center and len(unique_center)==1:
                labels_list[index] = int(unique_center[0])

    unique_labels = list(set(labels_list) - set([0]))
    # positive_loss = 0
    # negative_loss = 0 
    positive_loss = []
    negative_loss = []
    num_pos_pair = 0
    num_neg_pair = 0
    tripletloss = 0
    for batch_label in unique_labels:
        positive_mask_indices = []
        negative_mask_indices = []
        # torch_triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7, reduction='mean')
        for index in range(len(labels_list)):
            if labels_list[index] == batch_label:
                positive_mask_indices.append(index)
            elif labels_list[index] != 0:
                negative_mask_indices.append(index)
            # else:
            #     negative_index_mask.append(index)
        for hierarchy_level in range(len(mus)):
            if hierarchy_level==0 or hierarchy_level==1:
                from_index = 2**(5-hierarchy_level-1)-4
                # positive_z = [[mus[hierarchy_level][i][from_index:from_index+8, from_index:from_index+8],
                #                logvars[hierarchy_level][i][from_index:from_index+8, from_index:from_index+8]] for i in positive_index_mask]
                # negative_z = [[mus[hierarchy_level][i][from_index:from_index+8, from_index:from_index+8],
                #                logvars[hierarchy_level][i][from_index:from_index+8, from_index:from_index+8]] for i in negative_index_mask]
                positive_z = [mus[hierarchy_level][i][:,from_index:from_index+8, from_index:from_index+8] for i in positive_mask_indices]
                negative_z = [mus[hierarchy_level][i][:,from_index:from_index+8, from_index:from_index+8] for i in negative_mask_indices]
            else:
                # positive_z = [[mus[hierarchy_level][i], logvars[hierarchy_level][i]] for i in positive_index_mask]
                # negative_z = [[mus[hierarchy_level][i], logvars[hierarchy_level][i]] for i in negative_index_mask]
                positive_z = [mus[hierarchy_level][i][:] for i in positive_mask_indices]
                negative_z = [mus[hierarchy_level][i][:] for i in negative_mask_indices]
        
            if cl_mode == 'min max':
                res = [(a, b) for idx, a in enumerate(positive_z) for b in positive_z[idx + 1:]]
                # for (a,b) in itertools.combinations(collection, 2): TODO try out on a toy example, it might be faster and easier to read
                
                for (a,b) in res:
                    positive_loss.append(metric_ed(a,b))
                    num_pos_pair += 1
                
                for i in range(len(positive_z)):
                    for j in range(len(negative_z)):
                        negative_loss.append(metric_ed(positive_z[i],negative_z[j]))
                        num_neg_pair += 1
                        
                max_pos = 0
                min_neg = 0
                positive_loss = [a.max() for a in positive_loss]
                negative_loss = [b.min() for b in negative_loss]
                if num_pos_pair > 0:
                    max_pos = max(positive_loss)
                if num_neg_pair > 0:
                    min_neg = min(negative_loss)
                tripletloss += max_pos - min_neg + margin
    return tripletloss
    #         if cl_mode == 'triplet loss':
    #             for indx in range(len(positive_z)):
    #                 anchor = positive_z[indx][0] #index in batch, mu, first sample in batch
    #                 # positive_z shape number of samples in batch, 2, 8, 8, 32
    #                 tripletloss += triplet_loss(anchor, positive_z, negative_z)
    #                 # tripletloss += triplet_loss(anchor, positive_z, negative_z, torch_triplet_loss)

    #         if cl_mode == 'kl divergence': #TODO fix this conditions
    #             res = [(a, b) for idx, a in enumerate(positive_z) for b in positive_z[idx + 1:]]
    #             for (a,b) in res:
    #                 if cl_mode == 'kl divergence':
    #                     positive_loss += metric_kl(a,b)
    #                 elif cl_mode == 'cosine similarity':
    #                     positive_loss += metric_cs(a,b)
    #                 elif cl_mode == 'euclidean distance' or cl_mode == 'triplet and cosine':
    #                     positive_loss += euclidean_distance_based(a,b,target=1)
    #                 num_pos_pair += 1

    #             for i in range(len(positive_z)):
    #                 for j in range(len(negative_z)):
    #                     if cl_mode == 'kl divergence':
    #                         negative_loss += metric_kl(positive_z[i], negative_z[j])
    #                     elif cl_mode == 'cosine similarity':
    #                         negative_loss += metric_cs(positive_z[i], negative_z[j])
    #                     elif cl_mode == 'euclidean distance' or cl_mode == 'triplet and cosine':
    #                         negative_loss += euclidean_distance_based(positive_z[i], negative_z[j], target=0)
    #                     num_neg_pair += 1
    
    # if cl_mode == 'kl divergence':
    #     return positive_loss - torch.clip(negative_loss, max=1e+3)
    # elif cl_mode == 'cosine similarity':
    #     if num_neg_pair!=0 and num_pos_pair!=0:
    #         return negative_loss/num_neg_pair - positive_loss/num_pos_pair
    #     else:
    #         return negative_loss - positive_loss
    # elif cl_mode == 'euclidean distance':
    #     if num_neg_pair!=0 and num_pos_pair!=0:
    #         return  positive_loss/num_pos_pair + negative_loss/num_neg_pair
    #     else:
    #         return positive_loss + negative_loss
    # elif cl_mode == 'triplet loss':
    #     return tripletloss
    # elif cl_mode == 'triplet and cosine':
    #     if num_neg_pair!=0 and num_pos_pair!=0:
    #         return negative_loss/num_neg_pair - positive_loss/num_pos_pair + tripletloss
    #     else:
    #         return negative_loss - positive_loss + tripletloss

# def contrastive_loss(z, labels, margin=1.0):
#     # Compute pairwise distances
#     batch_size = z.size(0)
#     dist = torch.cdist(z, z, p=2)
    
#     # Create a mask for positive and negative pairs
#     labels = labels.unsqueeze(1)
#     positive_mask = labels == labels.T
#     negative_mask = ~positive_mask
    
#     # Positive pairs: minimize distance
#     positive_loss = torch.sum(positive_mask * dist)
    
#     # Negative pairs: maximize distance (with margin)
#     negative_loss = torch.sum(negative_mask * F.relu(margin - dist))
    
#     # Normalize by number of pairs
#     num_positive_pairs = positive_mask.sum()
#     num_negative_pairs = negative_mask.sum()
    
#     positive_loss /= num_positive_pairs
#     negative_loss /= num_negative_pairs
    
#     # Total contrastive loss
#     loss = positive_loss + negative_loss
#     return loss