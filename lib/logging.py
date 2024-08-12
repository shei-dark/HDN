from openTSNE import TSNE
import matplotlib.patches as patches
from matplotlib import pyplot
import torch
from tifffile import imread
import numpy as np
from scipy.stats import shapiro, anderson
from sklearn.metrics import silhouette_score, davies_bouldin_score
from enum import IntEnum
from scipy.spatial.distance import cdist
import seaborn as sns
from lib import plotting as p


patch_size = 64
centre_size = 4
n_channel = 32
hierarchy_level = 3
sample_size = 100
dimension = 43008
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class ClassType(IntEnum):

    UNCATEGORIZED = 0
    NUCLEUS = 1
    GRANULES = 2
    MITOCHONDRIA = 3
    

def log_all_plots(wandb, test_set, model):

    mu = []
    mus = np.array([])
    
    for class_t in ClassType:
        indices_list = test_set.patches_by_label[class_t.value]
        patches, clss, lbls = test_set[indices_list]
        for patch in patches:                          
            mu.extend(get_feature_maps(model, patch))
        mus = np.append(mus, mu).reshape(-1, dimension)
        mu = []
    for i in range(len(mus)):
        mus[i] = np.asarray(mus[i])

    # silhouette
    labels = np.array([0] * sample_size + [1] * sample_size + [2] * sample_size + [3] * sample_size)
    silhouette = silhouette_score(mus, labels)
    wandb.log({"silhouette": silhouette})

    # davies bouldin
    davies_bouldin = davies_bouldin_score(mus, labels)
    wandb.log({"davies_bouldin": davies_bouldin})

    normal = 0
    for dim in range(dimension):
        stat, p_value = shapiro(mus[:,dim])
        if p_value > 0.05:
            normal += 1
    wandb.log({"Shapiro": normal})
    normal = 0
    for dim in range(dimension):
        result = anderson(mus[:,dim], dist='norm')
        if result.statistic < result.critical_values[2]:
            normal += 1
    wandb.log({"Anderson": normal})

    distances = cdist(mus, mus, metric='euclidean')

    wandb.log({"Pairwise Distance Heatmap": wandb.Histogram(distances)})

    plt.clf()

    tsne = []
    tsne = TSNE(
        n_components=2,
        perplexity=50,
        random_state=42,
        learning_rate="auto",
        metric="cosine",
        n_iter=10000,
    ).fit(mus)
    bg_tsne = tsne[:sample_size]
    nuc_tsne = tsne[sample_size:2*sample_size]
    gra_tsne = tsne[2*sample_size:3*sample_size]
    mito_tsne = tsne[3*sample_size:]

    point_size = 10
    plt = p.plot_w_b(([bg_tsne, nuc_tsne, gra_tsne, mito_tsne], "TSNE",\
                    [([0]*sample_size,point_size), ([1]*sample_size,point_size), ([2]*sample_size,point_size),\
                    ([3]*sample_size,point_size)]),\
            plot_types=['scatter'],box_size=10
            )
    wandb.log({"TSNE": wandb.Image(plt)})
    plt.clf()

# Function to draw custom bracket annotations
def draw_custom_bracket(ax, start, end, label):
    mid = (start + end) / 2
    # Draw the bracket
    ax.plot([-0.05, -0.1], [start, start], color='black', lw=1.5)
    ax.plot([-0.1, -0.1], [start, end], color='black', lw=1.5)
    ax.plot([-0.1, -0.05], [end, end], color='black', lw=1.5)
    # Add the text label
    ax.text(-0.15, mid, label, va='center', ha='right', fontsize=12)


def get_normalized_tensor(img, model, device):
    test_images = torch.from_numpy(img.copy()).to(device)
    data_mean = model.data_mean
    data_std = model.data_std
    test_images = (test_images - data_mean) / data_std
    return test_images


def load_data(dir):
    return imread(dir)


def get_pred(model, z):
    model.mode_pred = True
    model.eval()
    with torch.no_grad():
        model.to(device)
        z = z.to(device=device, dtype=torch.float)
        z = z.reshape(1, 1, patch_size, patch_size)
        with torch.no_grad():
            sample = model(z, z, z, model_layers=[0, 1, 2])
            pred = sample["out_mean"]
    return pred


def get_kl(model, z):
    model.mode_pred = True
    model.eval()
    with torch.no_grad():
        model.to(device)
        z = z.to(device=device, dtype=torch.float)
        z = z.reshape(1, 1, patch_size, patch_size)
        with torch.no_grad():
            sample = model(z, z, z, model_layers=[0, 1, 2])
            kl = [
                sample["kl_loss"] / (16 * 21),
                sample["kl_spatial"][0],
                sample["kl_spatial"][1],
                sample["kl_spatial"][2],
            ]
    return kl

def get_feature_maps(model, patch):
    mus = get_mus(model, patch)
    feature_map = [mus[i].flatten() for i in range(len(mus))]
    feature_map = np.array(torch.cat(feature_map,dim=-1).cpu().numpy())
    return feature_map

def get_mus(model, patch):
        hierarchy_level = model.n_layers
        n_channels = model.z_dims
        model.mode_pred = True
        model.eval()
        device = model.device
        with torch.no_grad():
            patch = patch.to(device=device, dtype=torch.float)
            patch = patch.reshape(1, 1, patch_size, patch_size)
            with torch.no_grad():
                sample = model(patch, patch, patch, model_layers=[i for i in range(hierarchy_level)])
                mu = sample["mu"]
        return mu
