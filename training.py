import numpy as np
import math
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from torch.nn import init
from torch.optim.optimizer import Optimizer
import os
import glob
import random
from tifffile import imread
from matplotlib import pyplot as plt
from tqdm import tqdm

from boilerplate import boilerplate
from models.lvae import LadderVAE
import lib.utils as utils
import wandb
import random
from sklearn.manifold import TSNE
import matplotlib.patches as patches
from glob import glob

# import plotly.express as px

patch_size = 64
centre_size = 4
n_channel = 32
hierarchy_level = 3
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def train_network(
    model,
    lr,
    max_epochs,
    steps_per_epoch,
    train_loader,
    val_loader,
    # test_loader,
    virtual_batch,
    gaussian_noise_std,
    model_name,
    test_log_every=1000,
    directory_path="./",
    val_loss_patience=100,
    nrows=4,
    max_grad_norm=None,
    debug=False,
    project_name="",
    save_output=True,
    batch_size=8,
    cl_w=1e-7,
    kl_w=1,
):

    """Train Hierarchical DivNoising network.
    Parameters
    ----------
    model: Ladder VAE object
        Hierarchical DivNoising model.
    lr: float
        Learning rate
    max_epochs: int
        Number of epochs to train the model for.
    train_loader: PyTorch data loader
        Data loader for training set.
    val_loader: PyTorch data loader
        Data loader for validation set.
    test_loader: PyTorch data loader
        Data loader for test set.
    virtual_batch: int
        Virtual batch size for training
    gaussian_noise_std: float
        standard deviation of gaussian noise (required when 'noiseModel' is None).
    model_name: String
        Name of Hierarchical DivNoising model with which to save weights.
    test_log_every: int
        Number of training steps after which one test evaluation is performed.
    directory_path: String
        Path where the DivNoising weights to be saved.
    val_loss_patience: int
        Number of epoochs after which training should be terminated if validation loss doesn't improve by 1e-6.
    max_grad_norm: float
        Value to limit/clamp the gradients at.
    """
    wandb.login()
    if debug == False:
        use_wandb = True
    else:
        use_wandb = False

    if use_wandb:
        run = wandb.init(
        # Set the project where this run will be logged
        project=project_name,
        # Track hyperparameters and run metadata
        config={
            "learning_rate": lr,
            "epochs": max_epochs,
            "batch_size": batch_size,
            "contrastive_learning_weight": cl_w,
            "KLD weight": kl_w,
        },
        )

        run.config.update(dict(epochs=max_epochs))

    model_folder = directory_path + "model/"
    img_folder = directory_path + "imgs/"
    device = model.device
    optimizer, scheduler = boilerplate._make_optimizer_and_scheduler(model, lr, 0.0)
    loss_train_history = []
    reconstruction_loss_train_history = []
    kl_loss_train_history = []
    cl_loss_train_history = []
    loss_val_history = []
    running_loss = 0.0
    step_counter = 0
    epoch = 0
    cl_w = cl_w
    kl_w = kl_w

    patience_ = 0
    first_step = True
    data_dir = "/group/jug/Sheida/pancreatic beta cells/download/high_c1/contrastive/patches/testing/img/"
    # data_dir = "/localscratch/testing/img/"
    golgi = get_normalized_tensor(load_data(sorted(glob(data_dir+'class1/*.tif'))), model, device)
    mitochondria = get_normalized_tensor(load_data(sorted(glob(data_dir+'class2/*.tif'))), model, device)
    granule = get_normalized_tensor(load_data(sorted(glob(data_dir+'class3/*.tif'))), model, device)
    mask_dir = "/group/jug/Sheida/pancreatic beta cells/download/high_c1/contrastive/patches/testing/mask/"
    # mask_dir = "/localscratch/testing/mask/"
    golgi_mask = load_data(sorted(glob(mask_dir+'class1/*.tif')))
    mitochondria_mask = load_data(sorted(glob(mask_dir+'class2/*.tif')))
    granule_mask = load_data(sorted(glob((mask_dir+'class3/*.tif'))))
    class_type = [golgi, mitochondria, granule]
    masks = [golgi_mask, mitochondria_mask, granule_mask]

    try:
        os.makedirs(model_folder)
    except FileExistsError:
        # directory already exists
        pass

    try:
        os.makedirs(img_folder)
    except FileExistsError:
        # directory already exists
        pass

    seconds_last = time.time()

    while step_counter / steps_per_epoch < max_epochs:
        epoch = epoch + 1
        running_training_loss = []
        running_reconstruction_loss = []
        running_kl_loss = []
        running_cl_loss = []

        for batch_idx, (x, y) in enumerate(train_loader):
            step_counter = batch_idx
            x = x.unsqueeze(1)  # Remove for RGB
            x = x.to(device=device, dtype=torch.float)
            model.mode_pred = False
            model.train()
            optimizer.zero_grad()

            outputs = boilerplate.forward_pass(
                    x, y, device, model, gaussian_noise_std
                )
            
            recons_loss = outputs["recons_loss"]
            kl_loss = outputs["kl_loss"]
            cl_loss = outputs["cl_loss"]
            if model.contrastive_learning and not model.use_non_stochastic:
                loss = recons_loss + kl_w * kl_loss + cl_w * cl_loss
            elif model.use_non_stochastic:
                loss = recons_loss + cl_w * cl_loss
            else:
                loss = recons_loss + kl_w * kl_loss
            loss.backward()

            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=max_grad_norm
                )
            # Optimization step

            running_training_loss.append(loss.item())
            running_reconstruction_loss.append(recons_loss.item())
            running_kl_loss.append(kl_loss.item())
            running_cl_loss.append(cl_loss.item())
#
            optimizer.step()

            # first_step = False
        if step_counter % steps_per_epoch == steps_per_epoch-1:
            if True:

                to_print = "Epoch[{}/{}] Training Loss: {:.3f} Reconstruction Loss: {:.3f} KL Loss: {:.3f} CL Loss: {:.3f}"
                to_print = to_print.format(
                    epoch,
                    max_epochs,
                    np.mean(running_training_loss),
                    np.mean(running_reconstruction_loss),
                    np.mean(running_kl_loss),
                    np.mean(running_cl_loss),
                )
                if use_wandb:
                    run.log(
                            {
                                "epoch": epoch,
                                "inpainting_loss": np.mean(running_reconstruction_loss),
                                "kl_loss": np.mean(running_kl_loss)*kl_w,
                                "cl_loss": np.mean(running_cl_loss)*cl_w,
                                "loss": np.mean(running_training_loss),
                            }
                    )
                print(to_print)
                print("saving", model_folder + model_name + "_last_vae.net")
                torch.save(model, model_folder + model_name + "_last_vae.net")

                ### Save training losses
                loss_train_history.append(np.mean(running_training_loss))
                reconstruction_loss_train_history.append(
                    np.mean(running_reconstruction_loss)
                )
                kl_loss_train_history.append(np.mean(running_kl_loss))
                cl_loss_train_history.append(np.mean(running_cl_loss))
                np.save(model_folder + "train_loss.npy", np.array(loss_train_history))
                np.save(
                    model_folder + "train_reco_loss.npy",
                    np.array(reconstruction_loss_train_history),
                )
                np.save(
                    model_folder + "train_kl_loss.npy", np.array(kl_loss_train_history)
                )
                np.save(
                    model_folder + "train_cl_loss.npy", np.array(cl_loss_train_history)
                )

                ### Validation step
                running_validation_loss = []
                val_cl = []
                val_kl = []
                val_inpainting = []
                model.eval()
                with torch.no_grad():
                    for i, (x, y) in enumerate(val_loader):
                        x = x.unsqueeze(1)  # Remove for RGB
                        x = x.to(device=device, dtype=torch.float)
                        val_outputs = boilerplate.forward_pass(
                            x, y, device, model, gaussian_noise_std
                        )

                        val_recons_loss = val_outputs["recons_loss"]
                        val_kl_loss = val_outputs['kl_loss']
                        val_cl_loss = val_outputs["cl_loss"]
                        val_loss = val_recons_loss + kl_w * val_kl_loss + cl_w * val_cl_loss
                        

                        # val_loss = val_recons_loss + cl_w * val_cl_loss
                        running_validation_loss.append(val_loss)
                        val_cl.append(val_cl_loss*cl_w)
                        val_kl.append(val_kl_loss*kl_w)
                        val_inpainting.append(val_recons_loss)
                    
                    wandb.log({"val_inpainting_loss": torch.mean(torch.stack(val_inpainting)),
                                "val_kl_loss": torch.mean(torch.stack(val_kl)),
                                "val_cl_loss": torch.mean(torch.stack(val_cl)),
                                "val_loss": torch.mean(torch.stack(running_validation_loss))})

                    mu = []
                    mus = np.array([])
                    kl = [{'kl_total':[], 'kl_0':[], 'kl_1':[], 'kl_2':[]},
                           {'kl_total':[], 'kl_0':[], 'kl_1':[], 'kl_2':[]},
                           {'kl_total':[], 'kl_0':[], 'kl_1':[], 'kl_2':[]}]
                    for class_t in range(len(class_type)):
                        for i in tqdm(range(len(class_type[class_t]))):
                            mu.extend(get_mus(model,class_type[class_t][i]))
                            kl_losses = get_kl(model,class_type[class_t][i])
                            kl[class_t]['kl_total'].append(kl_losses[0])
                            kl[class_t]['kl_0'].append(kl_losses[1])
                            kl[class_t]['kl_1'].append(kl_losses[2])
                            kl[class_t]['kl_2'].append(kl_losses[3])
                        mus = np.append(mus, mu).reshape(-1, 96)
                        mu = []
                    for i in range(len(mus)):
                        mus[i] = np.asarray(mus[i])

                    # golgi
                    kl_total = torch.mean(torch.stack(kl[0]['kl_total']))
                    kl_first_layer = torch.mean(torch.stack(kl[0]['kl_0']))
                    kl_second_layer = torch.mean(torch.stack(kl[0]['kl_1']))
                    kl_third_layer = torch.mean(torch.stack(kl[0]['kl_2']))
                    wandb.log({"kl_total_golgi": kl_total, "kl_first_layer_golgi": kl_first_layer, "kl_second_layer_golgi": kl_second_layer, "kl_third_layer_golgi": kl_third_layer})
                    kl_total = torch.mean(torch.stack(kl[1]['kl_total']))
                    kl_first_layer = torch.mean(torch.stack(kl[1]['kl_0']))
                    kl_second_layer = torch.mean(torch.stack(kl[1]['kl_1']))
                    kl_third_layer = torch.mean(torch.stack(kl[1]['kl_2']))
                    wandb.log({"kl_total_mitochondria": kl_total, "kl_first_layer_mitochondria": kl_first_layer, "kl_second_layer_mitochondria": kl_second_layer, "kl_third_layer_mitochondria": kl_third_layer})
                    kl_total = torch.mean(torch.stack(kl[2]['kl_total']))
                    kl_first_layer = torch.mean(torch.stack(kl[2]['kl_0']))
                    kl_second_layer = torch.mean(torch.stack(kl[2]['kl_1']))
                    kl_third_layer = torch.mean(torch.stack(kl[2]['kl_2']))
                    wandb.log({"kl_total_granule": kl_total, "kl_first_layer_granule": kl_first_layer, "kl_second_layer_granule": kl_second_layer, "kl_third_layer_granule": kl_third_layer})



                    X_embedded = []
                    X_embedded = TSNE(n_components=2, random_state= 42, method="exact", learning_rate='auto', init='pca', metric='cosine', n_iter= 10000, n_iter_without_progress=500).fit_transform(mus)
                    mean_of_classes = []
                    mean_of_classes.append([X_embedded[:19].T[0].mean(), X_embedded[:19].T[1].mean()])
                    mean_of_classes.append([X_embedded[19:180].T[0].mean(), X_embedded[19:180].T[1].mean()])
                    mean_of_classes.append([X_embedded[180:].T[0].mean(), X_embedded[180:].T[1].mean()])

                    fig, ax = plt.subplots()
                    fig.set_figheight(10)
                    fig.set_figwidth(10)

                    ax.scatter(X_embedded[:19].T[0], X_embedded[:19].T[1], c='orange', s=10, label='Golgi', alpha=1, edgecolors='none')
                    ax.scatter(mean_of_classes[0][0], mean_of_classes[0][1], c='orange', s=50, label='Golgi mean', alpha=1, edgecolors='none')
                    ax.scatter(X_embedded[19:180].T[0], X_embedded[19:180].T[1], c='blue', s=10, label='Mitochondria', alpha=1, edgecolors='none')
                    ax.scatter(mean_of_classes[1][0], mean_of_classes[1][1], c='blue', s=50, label='Mitochondria mean', alpha=1, edgecolors='none')
                    ax.scatter(X_embedded[180:].T[0], X_embedded[180:].T[1], c='green', s=10, label='Granule', alpha=1, edgecolors='none')
                    ax.scatter(mean_of_classes[2][0], mean_of_classes[2][1], c='green', s=50, label='Granule mean', alpha=1, edgecolors='none')

                    wandb.log({"t-SNE": wandb.Image(plt)})
                    plt.clf()

                    dis_matrix = np.array([])
                    for i in range(543):
                        for j in range(543):
                            dis_matrix = np.append(dis_matrix, np.linalg.norm(mus[i] - mus[j]))

                    dis_matrix = dis_matrix.reshape(543,543)

                    fig, ax = plt.subplots()
                    im = ax.imshow(dis_matrix)
                    fig.tight_layout()
                    wandb.log({"heatmap": wandb.Image(plt)})
                    plt.clf()

                    i,j,k = np.random.randint(len(class_type[0])), np.random.randint(len(class_type[1])), np.random.randint(len(class_type[2]))
                    print(i,j,k)
                    golgi_sample = get_pred(model, class_type[0][i])
                    mitochondria_sample = get_pred(model, class_type[1][j])
                    granule_sample = get_pred(model, class_type[2][k])

                    fig, ax = plt.subplots(3, 3)
                    rect = patches.Rectangle((29.5, 29.5), 4, 4, linewidth=2, edgecolor='r', facecolor='none')
                    ax[0,0].imshow(class_type[0][i].cpu().numpy().reshape(64,64))
                    ax[0,0].set_title("Golgi sample GT")
                    ax[0,0].add_patch(rect)
                    rect = patches.Rectangle((29.5, 29.5), 4, 4, linewidth=2, edgecolor='r', facecolor='none')
                    ax[1,0].imshow(golgi_sample.cpu().numpy().reshape(64,64))
                    ax[1,0].set_title("Golgi sample Pred")
                    ax[1,0].add_patch(rect)
                    rect = patches.Rectangle((29.5, 29.5), 4, 4, linewidth=2, edgecolor='r', facecolor='none')
                    ax[0,1].imshow(class_type[1][j].cpu().numpy().reshape(64,64))
                    ax[0,1].set_title("Mitochondria sample GT")
                    ax[0,1].add_patch(rect)
                    rect = patches.Rectangle((29.5, 29.5), 4, 4, linewidth=2, edgecolor='r', facecolor='none')
                    ax[1,1].imshow(mitochondria_sample.cpu().numpy().reshape(64,64))
                    ax[1,1].set_title("Mitochondria sample Pred")
                    ax[1,1].add_patch(rect)
                    rect = patches.Rectangle((29.5, 29.5), 4, 4, linewidth=2, edgecolor='r', facecolor='none')
                    ax[0,2].imshow(class_type[2][k].cpu().numpy().reshape(64,64))
                    ax[0,2].set_title("Granule sample GT")
                    ax[0,2].add_patch(rect)
                    rect = patches.Rectangle((29.5, 29.5), 4, 4, linewidth=2, edgecolor='r', facecolor='none')
                    ax[1,2].imshow(granule_sample.cpu().numpy().reshape(64,64))
                    ax[1,2].set_title("Granule sample Pred")
                    ax[1,2].add_patch(rect)
                    rect = patches.Rectangle((29.5, 29.5), 4, 4, linewidth=2, edgecolor='r', facecolor='none')
                    ax[2,0].imshow(masks[0][i])
                    ax[2,0].set_title("Golgi mask")
                    ax[2,0].add_patch(rect)
                    rect = patches.Rectangle((29.5, 29.5), 4, 4, linewidth=2, edgecolor='r', facecolor='none')
                    ax[2,1].imshow(masks[1][j])
                    ax[2,1].set_title("Mitochondria mask")
                    ax[2,1].add_patch(rect)
                    rect = patches.Rectangle((29.5, 29.5), 4, 4, linewidth=2, edgecolor='r', facecolor='none')
                    ax[2,2].imshow(masks[2][k])
                    ax[2,2].set_title("Granule mask")
                    ax[2,2].add_patch(rect)
                    for a in ax:
                        for b in a:
                            b.set_xlim(30 - 5, 30 + 4 + 5)
                            b.set_ylim(30 + 4 + 5, 30 - 5)
                    
                    wandb.log({"samples closeup": wandb.Image(plt)})
                    plt.clf()

                    fig, ax = plt.subplots(3, 3)
                    rect = patches.Rectangle((29.5, 29.5), 4, 4, linewidth=2, edgecolor='r', facecolor='none')
                    ax[0,0].imshow(class_type[0][i].cpu().numpy().reshape(64,64))
                    ax[0,0].set_title("Golgi sample GT")
                    ax[0,0].add_patch(rect)
                    rect = patches.Rectangle((29.5, 29.5), 4, 4, linewidth=2, edgecolor='r', facecolor='none')
                    ax[1,0].imshow(golgi_sample.cpu().numpy().reshape(64,64))
                    ax[1,0].set_title("Golgi sample Pred")
                    ax[1,0].add_patch(rect)
                    rect = patches.Rectangle((29.5, 29.5), 4, 4, linewidth=2, edgecolor='r', facecolor='none')
                    ax[0,1].imshow(class_type[1][j].cpu().numpy().reshape(64,64))
                    ax[0,1].set_title("Mitochondria sample GT")
                    ax[0,1].add_patch(rect)
                    rect = patches.Rectangle((29.5, 29.5), 4, 4, linewidth=2, edgecolor='r', facecolor='none')
                    ax[1,1].imshow(mitochondria_sample.cpu().numpy().reshape(64,64))
                    ax[1,1].set_title("Mitochondria sample Pred")
                    ax[1,1].add_patch(rect)
                    rect = patches.Rectangle((29.5, 29.5), 4, 4, linewidth=2, edgecolor='r', facecolor='none')
                    ax[0,2].imshow(class_type[2][k].cpu().numpy().reshape(64,64))
                    ax[0,2].set_title("Granule sample GT")
                    ax[0,2].add_patch(rect)
                    rect = patches.Rectangle((29.5, 29.5), 4, 4, linewidth=2, edgecolor='r', facecolor='none')
                    ax[1,2].imshow(granule_sample.cpu().numpy().reshape(64,64))
                    ax[1,2].set_title("Granule sample Pred")
                    ax[1,2].add_patch(rect)
                    rect = patches.Rectangle((29.5, 29.5), 4, 4, linewidth=2, edgecolor='r', facecolor='none')
                    ax[2,0].imshow(masks[0][i])
                    ax[2,0].set_title("Golgi mask")
                    ax[2,0].add_patch(rect)
                    rect = patches.Rectangle((29.5, 29.5), 4, 4, linewidth=2, edgecolor='r', facecolor='none')
                    ax[2,1].imshow(masks[1][j])
                    ax[2,1].set_title("Mitochondria mask")
                    ax[2,1].add_patch(rect)
                    rect = patches.Rectangle((29.5, 29.5), 4, 4, linewidth=2, edgecolor='r', facecolor='none')
                    ax[2,2].imshow(masks[2][k])
                    ax[2,2].set_title("Granule mask")
                    ax[2,2].add_patch(rect)
                    
                    wandb.log({"samples": wandb.Image(plt)})
                    plt.clf()

                    mean_of_classes[0] = np.mean(mus[:19], axis=0)
                    mean_of_classes[1] = np.mean(mus[19:180], axis=0)
                    mean_of_classes[2] = np.mean(mus[180:], axis=0)

                    dis_to_mean = []
                    dis_to_mean.append([np.linalg.norm(mus[i] - mean_of_classes[0]) for i in range(19)])
                    dis_to_mean.append([np.linalg.norm(mus[i] - mean_of_classes[1]) for i in range(19)])
                    dis_to_mean.append([np.linalg.norm(mus[i] - mean_of_classes[2]) for i in range(19)])
                    labels = ['Golgi'] + ['Mitochondria'] + ['Granule']
                    plt.hist(dis_to_mean, bins=30, label=labels)
                    plt.title("dis of Golgi samples to mean of classes")
                    plt.legend()
                    wandb.log({"Golgi vs all means": wandb.Image(plt)})
                    plt.clf()

                    dis_to_mean = []
                    dis_to_mean.append([np.linalg.norm(mus[i] - mean_of_classes[0]) for i in range(19,180)])
                    dis_to_mean.append([np.linalg.norm(mus[i] - mean_of_classes[1]) for i in range(19,180)])
                    dis_to_mean.append([np.linalg.norm(mus[i] - mean_of_classes[2]) for i in range(19,180)])
                    plt.hist(dis_to_mean, bins=40, label=labels)
                    plt.title("dis of Mitochondria samples to mean of classes")
                    plt.legend()
                    wandb.log({"Mitochondria vs all means": wandb.Image(plt)})
                    plt.clf()

                    dis_to_mean = []
                    dis_to_mean.append([np.linalg.norm(mus[i] - mean_of_classes[0]) for i in range(180,543)])
                    dis_to_mean.append([np.linalg.norm(mus[i] - mean_of_classes[1]) for i in range(180,543)])
                    dis_to_mean.append([np.linalg.norm(mus[i] - mean_of_classes[2]) for i in range(180,543)])
                    plt.hist(dis_to_mean, bins=50, label=labels)
                    plt.title("dis of Granule samples to mean of classes")
                    plt.legend()
                    wandb.log({"Granule vs all means": wandb.Image(plt)})
                    plt.clf()

                    dis_to_mean = []
                    dis_to_mean.append([np.linalg.norm(mus[i] - mean_of_classes[0]) for i in range(19)])
                    dis_to_mean.append([np.linalg.norm(mus[i] - mean_of_classes[0]) for i in range(19,180)])
                    dis_to_mean.append([np.linalg.norm(mus[i] - mean_of_classes[0]) for i in range(180,543)])
                    plt.hist(dis_to_mean, bins=50, label=labels)
                    plt.title("dis mean Golgi to")
                    plt.legend()
                    wandb.log({"Golgi mean vs all": wandb.Image(plt)})
                    plt.clf()

                    dis_to_mean = []
                    dis_to_mean.append([np.linalg.norm(mus[i] - mean_of_classes[1]) for i in range(19)])
                    dis_to_mean.append([np.linalg.norm(mus[i] - mean_of_classes[1]) for i in range(19,180)])
                    dis_to_mean.append([np.linalg.norm(mus[i] - mean_of_classes[1]) for i in range(180,543)])
                    plt.hist(dis_to_mean, bins=50, label=labels)
                    plt.title("dis mean Mitochondria to")
                    plt.legend()
                    wandb.log({"Mitochondria mean vs all": wandb.Image(plt)})
                    plt.clf()

                    dis_to_mean = []
                    dis_to_mean.append([np.linalg.norm(mus[i] - mean_of_classes[2]) for i in range(19)])
                    dis_to_mean.append([np.linalg.norm(mus[i] - mean_of_classes[2]) for i in range(19,180)])
                    dis_to_mean.append([np.linalg.norm(mus[i] - mean_of_classes[2]) for i in range(180,543)])
                    plt.hist(dis_to_mean, bins=50, label=labels)
                    plt.title("dis mean Granule to")
                    plt.legend()
                    wandb.log({"Granule mean vs all": wandb.Image(plt)})
                    plt.clf()

                    # first layer
                    dis_to_mean = []
                    dis_to_mean.append([np.linalg.norm(mus[i][:32] - mean_of_classes[0][:32]) for i in range(19)])
                    dis_to_mean.append([np.linalg.norm(mus[i][:32] - mean_of_classes[0][:32]) for i in range(19,180)])
                    dis_to_mean.append([np.linalg.norm(mus[i][:32] - mean_of_classes[0][:32]) for i in range(180,543)])
                    plt.hist(dis_to_mean, bins=50, label=labels)
                    plt.title("dis mean Golgi 1st layer to")
                    plt.legend()
                    wandb.log({"Golgi mean vs all 1st layer": wandb.Image(plt)})
                    plt.clf()

                    # second layer
                    dis_to_mean = []
                    dis_to_mean.append([np.linalg.norm(mus[i][32:64] - mean_of_classes[0][32:64]) for i in range(19)])
                    dis_to_mean.append([np.linalg.norm(mus[i][32:64] - mean_of_classes[0][32:64]) for i in range(19,180)])
                    dis_to_mean.append([np.linalg.norm(mus[i][32:64] - mean_of_classes[0][32:64]) for i in range(180,543)])
                    plt.hist(dis_to_mean, bins=50, label=labels)
                    plt.title("dis mean Golgi 2nd layer to")
                    plt.legend()
                    wandb.log({"Golgi mean vs all 2nd layer": wandb.Image(plt)})
                    plt.clf()

                    # third layer
                    dis_to_mean = []
                    dis_to_mean.append([np.linalg.norm(mus[i][64:] - mean_of_classes[0][64:]) for i in range(19)])
                    dis_to_mean.append([np.linalg.norm(mus[i][64:] - mean_of_classes[0][64:]) for i in range(19,180)])
                    dis_to_mean.append([np.linalg.norm(mus[i][64:] - mean_of_classes[0][64:]) for i in range(180,543)])
                    plt.hist(dis_to_mean, bins=50, label=labels)
                    plt.title("dis mean Golgi 3rd layer to")
                    plt.legend()
                    wandb.log({"Golgi mean vs all 3rd layer": wandb.Image(plt)})
                    plt.clf()

                    # first layer
                    dis_to_mean = []
                    dis_to_mean.append([np.linalg.norm(mus[i][:32] - mean_of_classes[1][:32]) for i in range(19)])
                    dis_to_mean.append([np.linalg.norm(mus[i][:32] - mean_of_classes[1][:32]) for i in range(19,180)])
                    dis_to_mean.append([np.linalg.norm(mus[i][:32] - mean_of_classes[1][:32]) for i in range(180,543)])
                    plt.hist(dis_to_mean, bins=50, label=labels)
                    plt.title("dis mean Mitochondria 1st layer to")
                    plt.legend()
                    wandb.log({"Mitochondria mean vs all 1st layer": wandb.Image(plt)})
                    plt.clf()

                    # second layer
                    dis_to_mean = []
                    dis_to_mean.append([np.linalg.norm(mus[i][32:64] - mean_of_classes[1][32:64]) for i in range(19)])
                    dis_to_mean.append([np.linalg.norm(mus[i][32:64] - mean_of_classes[1][32:64]) for i in range(19,180)])
                    dis_to_mean.append([np.linalg.norm(mus[i][32:64] - mean_of_classes[1][32:64]) for i in range(180,543)])
                    plt.hist(dis_to_mean, bins=50, label=labels)
                    plt.title("dis mean Mitochondria 2nd layer to")
                    plt.legend()
                    wandb.log({"Mitochondria mean vs all 2nd layer": wandb.Image(plt)})
                    plt.clf()

                    # third layer
                    dis_to_mean = []
                    dis_to_mean.append([np.linalg.norm(mus[i][64:] - mean_of_classes[1][64:]) for i in range(19)])
                    dis_to_mean.append([np.linalg.norm(mus[i][64:] - mean_of_classes[1][64:]) for i in range(19,180)])
                    dis_to_mean.append([np.linalg.norm(mus[i][64:] - mean_of_classes[1][64:]) for i in range(180,543)])
                    plt.hist(dis_to_mean, bins=50, label=labels)
                    plt.title("dis mean Mitochondria 3rd layer to")
                    plt.legend()
                    wandb.log({"Mitochondria mean vs all 3rd layer": wandb.Image(plt)})
                    plt.clf()

                    # first layer
                    dis_to_mean = []
                    dis_to_mean.append([np.linalg.norm(mus[i][:32] - mean_of_classes[2][:32]) for i in range(19)])
                    dis_to_mean.append([np.linalg.norm(mus[i][:32] - mean_of_classes[2][:32]) for i in range(19,180)])
                    dis_to_mean.append([np.linalg.norm(mus[i][:32] - mean_of_classes[2][:32]) for i in range(180,543)])
                    plt.hist(dis_to_mean, bins=50, label=labels)
                    plt.title("dis mean Granule 1st layer to")
                    plt.legend()
                    wandb.log({"Granule mean vs all 1st layer": wandb.Image(plt)})
                    plt.clf()

                    # second layer
                    dis_to_mean = []
                    dis_to_mean.append([np.linalg.norm(mus[i][32:64] - mean_of_classes[2][32:64]) for i in range(19)])
                    dis_to_mean.append([np.linalg.norm(mus[i][32:64] - mean_of_classes[2][32:64]) for i in range(19,180)])
                    dis_to_mean.append([np.linalg.norm(mus[i][32:64] - mean_of_classes[2][32:64]) for i in range(180,543)])
                    plt.hist(dis_to_mean, bins=50, label=labels)
                    plt.title("dis mean Granule 2nd layer to")
                    plt.legend()
                    wandb.log({"Granule mean vs all 2nd layer": wandb.Image(plt)})
                    plt.clf()

                    # third layer
                    dis_to_mean = []
                    dis_to_mean.append([np.linalg.norm(mus[i][64:] - mean_of_classes[2][64:]) for i in range(19)])
                    dis_to_mean.append([np.linalg.norm(mus[i][64:] - mean_of_classes[2][64:]) for i in range(19,180)])
                    dis_to_mean.append([np.linalg.norm(mus[i][64:] - mean_of_classes[2][64:]) for i in range(180,543)])
                    plt.hist(dis_to_mean, bins=50, label=labels)
                    plt.title("dis mean Granule 3rd layer to")
                    plt.legend()
                    wandb.log({"Granule mean vs all 3rd layer": wandb.Image(plt)})
                    plt.clf()

                    
                total_epoch_loss_val = torch.mean(torch.stack(running_validation_loss))
                scheduler.step(total_epoch_loss_val)

                ### Save validation losses
                loss_val_history.append(total_epoch_loss_val.item())
                np.save(model_folder + "val_loss.npy", np.array(loss_val_history))

                if total_epoch_loss_val.item() < 1e-6 + np.min(loss_val_history):
                    patience_ = 0
                    print("saving", model_folder + model_name + "_best_vae.net")
                    torch.save(model, model_folder + model_name + "_best_vae.net")

                else:
                    patience_ += 1

                if epoch % 10 == 0:
                    torch.save(
                        model.state_dict(),
                        model_folder + "scheduled_weight_" + str(epoch) + ".net",
                    )

                print(
                    "Patience:",
                    patience_,
                    "Validation Loss:",
                    total_epoch_loss_val.item(),
                    "Min validation loss:",
                    np.min(loss_val_history),
                )

                seconds = time.time()
                secondsElapsed = np.float64(seconds - seconds_last)
                seconds_last = seconds
                remainingEps = (max_epochs + 1) - (epoch + 1)
                estRemainSeconds = (secondsElapsed) * (remainingEps)
                estRemainSecondsInt = int(secondsElapsed) * (remainingEps)
                print("Time for epoch: " + str(int(secondsElapsed)) + "seconds")

                print(
                    "Est remaining time: "
                    + str(datetime.timedelta(seconds=estRemainSecondsInt))
                    + " or "
                    + str(estRemainSecondsInt)
                    + " seconds"
                )

                print("----------------------------------------", flush=True)

                if patience_ == val_loss_patience:
                    return

            # break

def get_normalized_tensor(img,model,device):
    test_images = torch.from_numpy(img.copy()).to(device)
    data_mean = model.data_mean
    data_std = model.data_std
    test_images = (test_images-data_mean)/data_std
    return test_images

def load_data(dir):
    return imread(dir)

def get_pred(model, z):
    model.mode_pred=True
    model.eval()
    with torch.no_grad():
        model.to(device)
        z = z.to(device=device, dtype=torch.float)
        z = z.reshape(1,1,patch_size,patch_size)
        with torch.no_grad():
                sample = model(z, z, z,model_layers=[0,1,2])
                pred = sample['out_mean']
    return pred

def get_kl(model, z):
    model.mode_pred=True
    model.eval()
    with torch.no_grad():
        model.to(device)
        z = z.to(device=device, dtype=torch.float)
        z = z.reshape(1,1,patch_size,patch_size)
        with torch.no_grad():
                sample = model(z, z, z,model_layers=[0,1,2])
                kl = [sample['kl_loss']/(16*21), sample['kl_spatial'][0], sample['kl_spatial'][1], sample['kl_spatial'][2]]
    return kl

def get_mus(model, z):
    n_features = n_channel * hierarchy_level
    data = np.zeros((n_features,))
    model.mode_pred=True
    model.eval()
    with torch.no_grad():
        model.to(device)
        z = z.to(device=device, dtype=torch.float)
        z = z.reshape(1,1,patch_size,patch_size)
        with torch.no_grad():
                sample = model(z, z, z,model_layers=[0,1,2])
                mu = sample['mu']
                for i in range(hierarchy_level):
                    data[i*n_channel:(i+1)*n_channel] = get_mean_centre(mu, i)
                data = data.T.reshape(-1,n_features)
    return data

def get_mean_centre(x, i):
    if i == 3:
        return x[i][0].cpu().numpy().reshape(n_channel,-1).mean(-1)
    elif i == 4:
        return x[i][0].cpu().numpy().reshape(n_channel,-1).mean(-1)
    else:
        lower_bound = 2**(5-1-i)-int(centre_size/2)
        upper_bound = 2**(5-1-i)+int(centre_size/2)
        return x[i][0].cpu().numpy()[:,lower_bound:upper_bound,lower_bound:upper_bound].reshape(n_channel,-1).mean(-1)