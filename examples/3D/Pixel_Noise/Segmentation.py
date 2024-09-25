import sys
sys.path.append('../../../')
sys.path.append('/home/sheida.rahnamai/GIT/HDN/')
import torch
import numpy as np
from sklearn.cluster import HDBSCAN
import random
from tqdm import tqdm
# from lib.dataloader import CustomTestDataset
from boilerplate.dataloader import CustomDataset
import tifffile as tiff
import os
from boilerplate import boilerplate
from scipy.spatial.distance import cdist
from torch.nn.functional import unfold
from sklearn.cluster import KMeans

def get_closest(train, test, metric):

        distances = cdist(test, train, metric=metric)
        closest_index = np.argmin(distances, axis=1)
        return closest_index

def get_all_patches(image, patch_size, stride):
    """
    Efficiently get all patches from the image using unfold.
    """
    image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)  # Ensure float32
    patches = unfold(image_tensor, kernel_size=patch_size, stride=stride)
    patches = patches.permute(0, 2, 1).reshape(-1, patch_size, patch_size)
    return patches

def __main__():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    dist_metric = ['cosine']

    num_clusters = 4
    patch_size = 64
    sample_size = 500
    centre_size = 4
    n_channel = 32
    hierarchy_level = 3
    pad_size = (patch_size - centre_size) // 2
    model_dir = "/group/jug/Sheida/HVAE/3d/v01/model/"
    model = torch.load(model_dir+"HVAE_best_vae.net")

    data_dir = "/group/jug/Sheida/pancreatic beta cells/download/"

    data_mean = model.data_mean
    data_std = model.data_std

    One_test_image = ['high_c4']

    # Load test image
    test_img_path = os.path.join(data_dir, One_test_image[0], f"{One_test_image[0]}_source.tif")
    test_images = tiff.imread(test_img_path)

    # Print loaded test images paths
    print("Test image loaded from path:")
    print(test_img_path)

    # Load test ground truth images
    test_gt_path = os.path.join(data_dir, One_test_image[0], f"{One_test_image[0]}_gt.tif")
    test_ground_truth_image = tiff.imread(test_gt_path)

    model.mode_pred = True
    model.eval()
    device = model.device
    with torch.no_grad():
        train_batch = train_batch.to(device=device, dtype=torch.float)
        train_batch = train_batch.reshape(-1, 1, patch_size, patch_size)
        output = model(train_batch, model_layers=[i for i in range(hierarchy_level)])
        mu_train = torch.cat([output["mu"][i].view(sample_size * 4, -1) for i in range(hierarchy_level)], dim=1)
        mu_train_mean = torch.cat([torch.mean(mu_train[i*sample_size:(i+1)*sample_size], dim=0).unsqueeze(0) 
                                for i in range(4)], dim=0)
        mu_train_mean = np.array(mu_train_mean.cpu().numpy())

    #test data

    test_images = (test_images-data_mean)/data_std

    for test_index in tqdm(range(410, 500), desc='Test data'):

        test_slice = test_images[test_index]
        test_slice_lbl = test_ground_truth_image[test_index]
        all_patches = get_all_patches(test_slice, patch_size, 1).unsqueeze(1)
        test_batch_size = 400
        h, w = test_slice.shape
        new_h, new_w = h - patch_size + 1, w - patch_size + 1
        feature_maps = np.zeros((len(dist_metric),(new_h * new_w)))

        model.mode_pred = True
        model.eval()
        device = model.device
        index = 0 
        all_mus = np.zeros(((new_h * new_w), 43008))
        with torch.no_grad():
            all_patches = all_patches.to(device=device, dtype=torch.float)
            for start in tqdm(range(0, all_patches.shape[0], test_batch_size)):
                end = min(start + test_batch_size, all_patches.shape[0])
                output = model(all_patches[start:end], model_layers=[i for i in range(hierarchy_level)])
                mu_test = torch.cat([output["mu"][i].view(end - start, -1) for i in range(hierarchy_level)], dim=1)
                mu_test = np.array(mu_test.cpu().numpy())
                all_mus[index:index + test_batch_size] = mu_test
                for m in range(len(dist_metric)):
                    feature_maps[m][index:index + test_batch_size] = get_closest(mu_train_mean, mu_test, metric=dist_metric[m])
                index += test_batch_size

        feature_maps = feature_maps.reshape(len(dist_metric), new_h, new_w)
        # ['euclidean', 'cosine', 'correlation', 'seuclidean']
        for d in range(len(dist_metric)):
            tiff.imwrite(f"{data_dir}/seg/{dist_metric[d]}/seg{test_index}.tif", feature_maps[d])

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(all_mus)
        clusters = cluster_labels.reshape(new_h, new_w)
        tiff.imwrite(f"{data_dir}/seg/clusters{num_clusters}/seg{test_index}.tif", clusters)
