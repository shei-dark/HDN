import sys
sys.path.append('../../../')
import torch
from matplotlib import pyplot as plt
from tifffile import imread, imsave
from sklearn.cluster import KMeans
import numpy as np
from glob import glob

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

path="/group/jug/Sheida/maester_data/download/high_c1/"
patch_size = 64
# model = torch.load("/group/jug/Sheida/HDN models/*.net")
model = torch.load("/home/sheida.rahnamai/GIT/HDN/examples/Pixel_Noise/Convallaria/Trained_model/model/HDN Muller_best_vae.net")
model.mode_pred=True
model.eval()
model.to(device)

files = glob(path+"data/test/*.tif")

img_height,img_width = 699, 760
for i in files:
    img = imread(i)
    img_t = get_normalized_tensor(img,model,device)
    image_sample = img_t.view(1,1,img_height,img_width)
    image_sample = image_sample.to(device=device, dtype=torch.float)
    with torch.no_grad():
        sample = model(image_sample)
        for idx in range(len(sample['mu'])):
            mu = sample['mu'][idx][0].cpu().numpy()
            output = clustering(idx, mu)
            imsave(path+"label/"+str(idx+1)+"/"+i[i.rfind("/"):i.rfind(".")]+".tif", output.reshape(22*(2**(4-idx)), 24*(2**(4-idx))))

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
    test_images = torch.from_numpy(img.copy()).to(device)
    data_mean = model.data_mean
    data_std = model.data_std
    test_images = (test_images-data_mean)/data_std
    return test_images

def clustering(idx, mu):
    slice_features_mu = mu.astype(float)
    feature_flatten_mu = slice_features_mu.reshape(32, -1).T
    # feature_flatten_mu = slice_features_mu.reshape(32*(2**idx), -1).T
    # if idx == 0 or idx == 1 or idx == 2:
    #     feature_flatten_mu = slice_features_mu.reshape(32, -1).T
    # elif idx == 3 or idx == 4:
    #     feature_flatten_mu = slice_features_mu.reshape(64, -1).T
    K_CENTRE = 2
    kmeans_mu = KMeans(
        n_clusters=K_CENTRE, init='k-means++', n_init=10,
        max_iter=1000, random_state=777
    )
    kmeans_mu.fit(feature_flatten_mu)
    labels_mu = kmeans_mu.predict(feature_flatten_mu)
    return labels_mu