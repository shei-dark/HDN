from sklearn.manifold import TSNE
import matplotlib.patches as patches
from matplotlib import pyplot as plt
import torch
from tifffile import imread
import numpy as np
from scipy.stats import shapiro, anderson
from sklearn.metrics import silhouette_score, davies_bouldin_score

patch_size = 64
centre_size = 4
n_channel = 32
hierarchy_level = 3
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def log_all_plots(wandb, class_type, model, masks):

    mu = []
    mus = np.array([])
    kl = [
        {"kl_total": [], "kl_0": [], "kl_1": [], "kl_2": []},
        {"kl_total": [], "kl_0": [], "kl_1": [], "kl_2": []},
        {"kl_total": [], "kl_0": [], "kl_1": [], "kl_2": []},
    ]
    for class_t in range(len(class_type)):
        for i in range(len(class_type[class_t])):
            mu.extend(get_mus(model, class_type[class_t][i]))
            kl_losses = get_kl(model, class_type[class_t][i])
            kl[class_t]["kl_total"].append(kl_losses[0])
            kl[class_t]["kl_0"].append(kl_losses[1])
            kl[class_t]["kl_1"].append(kl_losses[2])
            kl[class_t]["kl_2"].append(kl_losses[3])
        mus = np.append(mus, mu).reshape(-1, 96)
        mu = []
    for i in range(len(mus)):
        mus[i] = np.asarray(mus[i])

    # silhouette
    labels = np.array([0] * 19 + [1] * 161 + [2] * 363)
    silhouette = silhouette_score(mus, labels)
    wandb.log({"silhouette": silhouette})

    # davies bouldin
    davies_bouldin = davies_bouldin_score(mus, labels)
    wandb.log({"davies_bouldin": davies_bouldin})

    normal = 0
    for dim in range(96):
        stat, p_value = shapiro(mus[:,dim])
        if p_value > 0.05:
            normal += 1
    wandb.log({"Shapiro": normal})
    normal = 0
    for dim in range(96):
        result = anderson(mus[:,dim], dist='norm')
        if result.statistic < result.critical_values[2]:
            normal += 1
    wandb.log({"Anderson": normal})

    # golgi
    kl_total = torch.mean(torch.stack(kl[0]["kl_total"]))
    kl_first_layer = torch.mean(torch.stack(kl[0]["kl_0"]))
    kl_second_layer = torch.mean(torch.stack(kl[0]["kl_1"]))
    kl_third_layer = torch.mean(torch.stack(kl[0]["kl_2"]))
    wandb.log(
        {
            "kl_total_golgi": kl_total,
            "kl_first_layer_golgi": kl_first_layer,
            "kl_second_layer_golgi": kl_second_layer,
            "kl_third_layer_golgi": kl_third_layer,
        }
    )
    kl_total = torch.mean(torch.stack(kl[1]["kl_total"]))
    kl_first_layer = torch.mean(torch.stack(kl[1]["kl_0"]))
    kl_second_layer = torch.mean(torch.stack(kl[1]["kl_1"]))
    kl_third_layer = torch.mean(torch.stack(kl[1]["kl_2"]))
    wandb.log(
        {
            "kl_total_mitochondria": kl_total,
            "kl_first_layer_mitochondria": kl_first_layer,
            "kl_second_layer_mitochondria": kl_second_layer,
            "kl_third_layer_mitochondria": kl_third_layer,
        }
    )
    kl_total = torch.mean(torch.stack(kl[2]["kl_total"]))
    kl_first_layer = torch.mean(torch.stack(kl[2]["kl_0"]))
    kl_second_layer = torch.mean(torch.stack(kl[2]["kl_1"]))
    kl_third_layer = torch.mean(torch.stack(kl[2]["kl_2"]))
    wandb.log(
        {
            "kl_total_granule": kl_total,
            "kl_first_layer_granule": kl_first_layer,
            "kl_second_layer_granule": kl_second_layer,
            "kl_third_layer_granule": kl_third_layer,
        }
    )

    X_embedded = []
    X_embedded = TSNE(
        n_components=2,
        random_state=42,
        method="exact",
        learning_rate="auto",
        init="pca",
        metric="cosine",
        n_iter=10000,
        n_iter_without_progress=500,
    ).fit_transform(mus)
    mean_of_classes = []
    mean_of_classes.append([X_embedded[:19].T[0].mean(), X_embedded[:19].T[1].mean()])
    mean_of_classes.append(
        [X_embedded[19:180].T[0].mean(), X_embedded[19:180].T[1].mean()]
    )
    mean_of_classes.append([X_embedded[180:].T[0].mean(), X_embedded[180:].T[1].mean()])

    fig, ax = plt.subplots()
    fig.set_figheight(10)
    fig.set_figwidth(10)

    ax.scatter(
        X_embedded[:19].T[0],
        X_embedded[:19].T[1],
        c="orange",
        s=10,
        label="Golgi",
        alpha=1,
        edgecolors="none",
    )
    ax.scatter(
        mean_of_classes[0][0],
        mean_of_classes[0][1],
        c="orange",
        s=50,
        label="Golgi mean",
        alpha=1,
        edgecolors="none",
    )
    ax.scatter(
        X_embedded[19:180].T[0],
        X_embedded[19:180].T[1],
        c="blue",
        s=10,
        label="Mitochondria",
        alpha=1,
        edgecolors="none",
    )
    ax.scatter(
        mean_of_classes[1][0],
        mean_of_classes[1][1],
        c="blue",
        s=50,
        label="Mitochondria mean",
        alpha=1,
        edgecolors="none",
    )
    ax.scatter(
        X_embedded[180:].T[0],
        X_embedded[180:].T[1],
        c="green",
        s=10,
        label="Granule",
        alpha=1,
        edgecolors="none",
    )
    ax.scatter(
        mean_of_classes[2][0],
        mean_of_classes[2][1],
        c="green",
        s=50,
        label="Granule mean",
        alpha=1,
        edgecolors="none",
    )

    wandb.log({"t-SNE": wandb.Image(plt)})
    plt.clf()

    dis_matrix = np.array([])
    for i in range(543):
        for j in range(543):
            dis_matrix = np.append(dis_matrix, np.linalg.norm(mus[i] - mus[j]))

    dis_matrix = dis_matrix.reshape(543, 543)

    fig, ax = plt.subplots()
    im = ax.imshow(dis_matrix)
    fig.tight_layout()
    wandb.log({"heatmap": wandb.Image(plt)})
    plt.clf()

    i, j, k = (
        np.random.randint(len(class_type[0])),
        np.random.randint(len(class_type[1])),
        np.random.randint(len(class_type[2])),
    )
    print(i, j, k)
    golgi_sample = get_pred(model, class_type[0][i])
    mitochondria_sample = get_pred(model, class_type[1][j])
    granule_sample = get_pred(model, class_type[2][k])

    fig, ax = plt.subplots(3, 3)
    rect = patches.Rectangle(
        (29.5, 29.5), 4, 4, linewidth=2, edgecolor="r", facecolor="none"
    )
    ax[0, 0].imshow(class_type[0][i].cpu().numpy().reshape(64, 64))
    ax[0, 0].set_title("Golgi sample GT")
    ax[0, 0].add_patch(rect)
    rect = patches.Rectangle(
        (29.5, 29.5), 4, 4, linewidth=2, edgecolor="r", facecolor="none"
    )
    ax[1, 0].imshow(golgi_sample.cpu().numpy().reshape(64, 64))
    ax[1, 0].set_title("Golgi sample Pred")
    ax[1, 0].add_patch(rect)
    rect = patches.Rectangle(
        (29.5, 29.5), 4, 4, linewidth=2, edgecolor="r", facecolor="none"
    )
    ax[0, 1].imshow(class_type[1][j].cpu().numpy().reshape(64, 64))
    ax[0, 1].set_title("Mitochondria sample GT")
    ax[0, 1].add_patch(rect)
    rect = patches.Rectangle(
        (29.5, 29.5), 4, 4, linewidth=2, edgecolor="r", facecolor="none"
    )
    ax[1, 1].imshow(mitochondria_sample.cpu().numpy().reshape(64, 64))
    ax[1, 1].set_title("Mitochondria sample Pred")
    ax[1, 1].add_patch(rect)
    rect = patches.Rectangle(
        (29.5, 29.5), 4, 4, linewidth=2, edgecolor="r", facecolor="none"
    )
    ax[0, 2].imshow(class_type[2][k].cpu().numpy().reshape(64, 64))
    ax[0, 2].set_title("Granule sample GT")
    ax[0, 2].add_patch(rect)
    rect = patches.Rectangle(
        (29.5, 29.5), 4, 4, linewidth=2, edgecolor="r", facecolor="none"
    )
    ax[1, 2].imshow(granule_sample.cpu().numpy().reshape(64, 64))
    ax[1, 2].set_title("Granule sample Pred")
    ax[1, 2].add_patch(rect)
    rect = patches.Rectangle(
        (29.5, 29.5), 4, 4, linewidth=2, edgecolor="r", facecolor="none"
    )
    ax[2, 0].imshow(masks[0][i])
    ax[2, 0].set_title("Golgi mask")
    ax[2, 0].add_patch(rect)
    rect = patches.Rectangle(
        (29.5, 29.5), 4, 4, linewidth=2, edgecolor="r", facecolor="none"
    )
    ax[2, 1].imshow(masks[1][j])
    ax[2, 1].set_title("Mitochondria mask")
    ax[2, 1].add_patch(rect)
    rect = patches.Rectangle(
        (29.5, 29.5), 4, 4, linewidth=2, edgecolor="r", facecolor="none"
    )
    ax[2, 2].imshow(masks[2][k])
    ax[2, 2].set_title("Granule mask")
    ax[2, 2].add_patch(rect)
    for a in ax:
        for b in a:
            b.set_xlim(30 - 5, 30 + 4 + 5)
            b.set_ylim(30 + 4 + 5, 30 - 5)

    wandb.log({"samples closeup": wandb.Image(plt)})
    plt.clf()

    fig, ax = plt.subplots(3, 3)
    rect = patches.Rectangle(
        (29.5, 29.5), 4, 4, linewidth=2, edgecolor="r", facecolor="none"
    )
    ax[0, 0].imshow(class_type[0][i].cpu().numpy().reshape(64, 64))
    ax[0, 0].set_title("Golgi sample GT")
    ax[0, 0].add_patch(rect)
    rect = patches.Rectangle(
        (29.5, 29.5), 4, 4, linewidth=2, edgecolor="r", facecolor="none"
    )
    ax[1, 0].imshow(golgi_sample.cpu().numpy().reshape(64, 64))
    ax[1, 0].set_title("Golgi sample Pred")
    ax[1, 0].add_patch(rect)
    rect = patches.Rectangle(
        (29.5, 29.5), 4, 4, linewidth=2, edgecolor="r", facecolor="none"
    )
    ax[0, 1].imshow(class_type[1][j].cpu().numpy().reshape(64, 64))
    ax[0, 1].set_title("Mitochondria sample GT")
    ax[0, 1].add_patch(rect)
    rect = patches.Rectangle(
        (29.5, 29.5), 4, 4, linewidth=2, edgecolor="r", facecolor="none"
    )
    ax[1, 1].imshow(mitochondria_sample.cpu().numpy().reshape(64, 64))
    ax[1, 1].set_title("Mitochondria sample Pred")
    ax[1, 1].add_patch(rect)
    rect = patches.Rectangle(
        (29.5, 29.5), 4, 4, linewidth=2, edgecolor="r", facecolor="none"
    )
    ax[0, 2].imshow(class_type[2][k].cpu().numpy().reshape(64, 64))
    ax[0, 2].set_title("Granule sample GT")
    ax[0, 2].add_patch(rect)
    rect = patches.Rectangle(
        (29.5, 29.5), 4, 4, linewidth=2, edgecolor="r", facecolor="none"
    )
    ax[1, 2].imshow(granule_sample.cpu().numpy().reshape(64, 64))
    ax[1, 2].set_title("Granule sample Pred")
    ax[1, 2].add_patch(rect)
    rect = patches.Rectangle(
        (29.5, 29.5), 4, 4, linewidth=2, edgecolor="r", facecolor="none"
    )
    ax[2, 0].imshow(masks[0][i])
    ax[2, 0].set_title("Golgi mask")
    ax[2, 0].add_patch(rect)
    rect = patches.Rectangle(
        (29.5, 29.5), 4, 4, linewidth=2, edgecolor="r", facecolor="none"
    )
    ax[2, 1].imshow(masks[1][j])
    ax[2, 1].set_title("Mitochondria mask")
    ax[2, 1].add_patch(rect)
    rect = patches.Rectangle(
        (29.5, 29.5), 4, 4, linewidth=2, edgecolor="r", facecolor="none"
    )
    ax[2, 2].imshow(masks[2][k])
    ax[2, 2].set_title("Granule mask")
    ax[2, 2].add_patch(rect)

    wandb.log({"samples": wandb.Image(plt)})
    plt.clf()

    mean_of_classes[0] = np.mean(mus[:19], axis=0)
    mean_of_classes[1] = np.mean(mus[19:180], axis=0)
    mean_of_classes[2] = np.mean(mus[180:], axis=0)

    dis_to_mean = []
    dis_to_mean.append([np.linalg.norm(mus[i] - mean_of_classes[0]) for i in range(19)])
    dis_to_mean.append([np.linalg.norm(mus[i] - mean_of_classes[1]) for i in range(19)])
    dis_to_mean.append([np.linalg.norm(mus[i] - mean_of_classes[2]) for i in range(19)])
    labels = ["Golgi"] + ["Mitochondria"] + ["Granule"]
    plt.hist(dis_to_mean, bins=30, label=labels)
    plt.title("dis of Golgi samples to mean of classes")
    plt.legend()
    wandb.log({"Golgi vs all means": wandb.Image(plt)})
    plt.clf()

    dis_to_mean = []
    dis_to_mean.append(
        [np.linalg.norm(mus[i] - mean_of_classes[0]) for i in range(19, 180)]
    )
    dis_to_mean.append(
        [np.linalg.norm(mus[i] - mean_of_classes[1]) for i in range(19, 180)]
    )
    dis_to_mean.append(
        [np.linalg.norm(mus[i] - mean_of_classes[2]) for i in range(19, 180)]
    )
    plt.hist(dis_to_mean, bins=40, label=labels)
    plt.title("dis of Mitochondria samples to mean of classes")
    plt.legend()
    wandb.log({"Mitochondria vs all means": wandb.Image(plt)})
    plt.clf()

    dis_to_mean = []
    dis_to_mean.append(
        [np.linalg.norm(mus[i] - mean_of_classes[0]) for i in range(180, 543)]
    )
    dis_to_mean.append(
        [np.linalg.norm(mus[i] - mean_of_classes[1]) for i in range(180, 543)]
    )
    dis_to_mean.append(
        [np.linalg.norm(mus[i] - mean_of_classes[2]) for i in range(180, 543)]
    )
    plt.hist(dis_to_mean, bins=50, label=labels)
    plt.title("dis of Granule samples to mean of classes")
    plt.legend()
    wandb.log({"Granule vs all means": wandb.Image(plt)})
    plt.clf()

    dis_to_mean = []
    dis_to_mean.append([np.linalg.norm(mus[i] - mean_of_classes[0]) for i in range(19)])
    dis_to_mean.append(
        [np.linalg.norm(mus[i] - mean_of_classes[0]) for i in range(19, 180)]
    )
    dis_to_mean.append(
        [np.linalg.norm(mus[i] - mean_of_classes[0]) for i in range(180, 543)]
    )
    plt.hist(dis_to_mean, bins=50, label=labels)
    plt.title("dis mean Golgi to")
    plt.legend()
    wandb.log({"Golgi mean vs all": wandb.Image(plt)})
    plt.clf()

    dis_to_mean = []
    dis_to_mean.append([np.linalg.norm(mus[i] - mean_of_classes[1]) for i in range(19)])
    dis_to_mean.append(
        [np.linalg.norm(mus[i] - mean_of_classes[1]) for i in range(19, 180)]
    )
    dis_to_mean.append(
        [np.linalg.norm(mus[i] - mean_of_classes[1]) for i in range(180, 543)]
    )
    plt.hist(dis_to_mean, bins=50, label=labels)
    plt.title("dis mean Mitochondria to")
    plt.legend()
    wandb.log({"Mitochondria mean vs all": wandb.Image(plt)})
    plt.clf()

    dis_to_mean = []
    dis_to_mean.append([np.linalg.norm(mus[i] - mean_of_classes[2]) for i in range(19)])
    dis_to_mean.append(
        [np.linalg.norm(mus[i] - mean_of_classes[2]) for i in range(19, 180)]
    )
    dis_to_mean.append(
        [np.linalg.norm(mus[i] - mean_of_classes[2]) for i in range(180, 543)]
    )
    plt.hist(dis_to_mean, bins=50, label=labels)
    plt.title("dis mean Granule to")
    plt.legend()
    wandb.log({"Granule mean vs all": wandb.Image(plt)})
    plt.clf()

    # first layer
    dis_to_mean = []
    dis_to_mean.append(
        [np.linalg.norm(mus[i][:32] - mean_of_classes[0][:32]) for i in range(19)]
    )
    dis_to_mean.append(
        [np.linalg.norm(mus[i][:32] - mean_of_classes[0][:32]) for i in range(19, 180)]
    )
    dis_to_mean.append(
        [np.linalg.norm(mus[i][:32] - mean_of_classes[0][:32]) for i in range(180, 543)]
    )
    plt.hist(dis_to_mean, bins=50, label=labels)
    plt.title("dis mean Golgi 1st layer to")
    plt.legend()
    wandb.log({"Golgi mean vs all 1st layer": wandb.Image(plt)})
    plt.clf()

    # second layer
    dis_to_mean = []
    dis_to_mean.append(
        [np.linalg.norm(mus[i][32:64] - mean_of_classes[0][32:64]) for i in range(19)]
    )
    dis_to_mean.append(
        [
            np.linalg.norm(mus[i][32:64] - mean_of_classes[0][32:64])
            for i in range(19, 180)
        ]
    )
    dis_to_mean.append(
        [
            np.linalg.norm(mus[i][32:64] - mean_of_classes[0][32:64])
            for i in range(180, 543)
        ]
    )
    plt.hist(dis_to_mean, bins=50, label=labels)
    plt.title("dis mean Golgi 2nd layer to")
    plt.legend()
    wandb.log({"Golgi mean vs all 2nd layer": wandb.Image(plt)})
    plt.clf()

    # third layer
    dis_to_mean = []
    dis_to_mean.append(
        [np.linalg.norm(mus[i][64:] - mean_of_classes[0][64:]) for i in range(19)]
    )
    dis_to_mean.append(
        [np.linalg.norm(mus[i][64:] - mean_of_classes[0][64:]) for i in range(19, 180)]
    )
    dis_to_mean.append(
        [np.linalg.norm(mus[i][64:] - mean_of_classes[0][64:]) for i in range(180, 543)]
    )
    plt.hist(dis_to_mean, bins=50, label=labels)
    plt.title("dis mean Golgi 3rd layer to")
    plt.legend()
    wandb.log({"Golgi mean vs all 3rd layer": wandb.Image(plt)})
    plt.clf()

    # first layer
    dis_to_mean = []
    dis_to_mean.append(
        [np.linalg.norm(mus[i][:32] - mean_of_classes[1][:32]) for i in range(19)]
    )
    dis_to_mean.append(
        [np.linalg.norm(mus[i][:32] - mean_of_classes[1][:32]) for i in range(19, 180)]
    )
    dis_to_mean.append(
        [np.linalg.norm(mus[i][:32] - mean_of_classes[1][:32]) for i in range(180, 543)]
    )
    plt.hist(dis_to_mean, bins=50, label=labels)
    plt.title("dis mean Mitochondria 1st layer to")
    plt.legend()
    wandb.log({"Mitochondria mean vs all 1st layer": wandb.Image(plt)})
    plt.clf()

    # second layer
    dis_to_mean = []
    dis_to_mean.append(
        [np.linalg.norm(mus[i][32:64] - mean_of_classes[1][32:64]) for i in range(19)]
    )
    dis_to_mean.append(
        [
            np.linalg.norm(mus[i][32:64] - mean_of_classes[1][32:64])
            for i in range(19, 180)
        ]
    )
    dis_to_mean.append(
        [
            np.linalg.norm(mus[i][32:64] - mean_of_classes[1][32:64])
            for i in range(180, 543)
        ]
    )
    plt.hist(dis_to_mean, bins=50, label=labels)
    plt.title("dis mean Mitochondria 2nd layer to")
    plt.legend()
    wandb.log({"Mitochondria mean vs all 2nd layer": wandb.Image(plt)})
    plt.clf()

    # third layer
    dis_to_mean = []
    dis_to_mean.append(
        [np.linalg.norm(mus[i][64:] - mean_of_classes[1][64:]) for i in range(19)]
    )
    dis_to_mean.append(
        [np.linalg.norm(mus[i][64:] - mean_of_classes[1][64:]) for i in range(19, 180)]
    )
    dis_to_mean.append(
        [np.linalg.norm(mus[i][64:] - mean_of_classes[1][64:]) for i in range(180, 543)]
    )
    plt.hist(dis_to_mean, bins=50, label=labels)
    plt.title("dis mean Mitochondria 3rd layer to")
    plt.legend()
    wandb.log({"Mitochondria mean vs all 3rd layer": wandb.Image(plt)})
    plt.clf()

    # first layer
    dis_to_mean = []
    dis_to_mean.append(
        [np.linalg.norm(mus[i][:32] - mean_of_classes[2][:32]) for i in range(19)]
    )
    dis_to_mean.append(
        [np.linalg.norm(mus[i][:32] - mean_of_classes[2][:32]) for i in range(19, 180)]
    )
    dis_to_mean.append(
        [np.linalg.norm(mus[i][:32] - mean_of_classes[2][:32]) for i in range(180, 543)]
    )
    plt.hist(dis_to_mean, bins=50, label=labels)
    plt.title("dis mean Granule 1st layer to")
    plt.legend()
    wandb.log({"Granule mean vs all 1st layer": wandb.Image(plt)})
    plt.clf()

    # second layer
    dis_to_mean = []
    dis_to_mean.append(
        [np.linalg.norm(mus[i][32:64] - mean_of_classes[2][32:64]) for i in range(19)]
    )
    dis_to_mean.append(
        [
            np.linalg.norm(mus[i][32:64] - mean_of_classes[2][32:64])
            for i in range(19, 180)
        ]
    )
    dis_to_mean.append(
        [
            np.linalg.norm(mus[i][32:64] - mean_of_classes[2][32:64])
            for i in range(180, 543)
        ]
    )
    plt.hist(dis_to_mean, bins=50, label=labels)
    plt.title("dis mean Granule 2nd layer to")
    plt.legend()
    wandb.log({"Granule mean vs all 2nd layer": wandb.Image(plt)})
    plt.clf()

    # third layer
    dis_to_mean = []
    dis_to_mean.append(
        [np.linalg.norm(mus[i][64:] - mean_of_classes[2][64:]) for i in range(19)]
    )
    dis_to_mean.append(
        [np.linalg.norm(mus[i][64:] - mean_of_classes[2][64:]) for i in range(19, 180)]
    )
    dis_to_mean.append(
        [np.linalg.norm(mus[i][64:] - mean_of_classes[2][64:]) for i in range(180, 543)]
    )
    plt.hist(dis_to_mean, bins=50, label=labels)
    plt.title("dis mean Granule 3rd layer to")
    plt.legend()
    wandb.log({"Granule mean vs all 3rd layer": wandb.Image(plt)})
    plt.clf()


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


def get_mus(model, z):
    n_features = n_channel * hierarchy_level
    data = np.zeros((n_features,))
    model.mode_pred = True
    model.eval()
    with torch.no_grad():
        model.to(device)
        z = z.to(device=device, dtype=torch.float)
        z = z.reshape(1, 1, patch_size, patch_size)
        with torch.no_grad():
            sample = model(z, z, z, model_layers=[0, 1, 2])
            mu = sample["mu"]
            for i in range(hierarchy_level):
                data[i * n_channel : (i + 1) * n_channel] = get_mean_centre(mu, i)
            data = data.T.reshape(-1, n_features)
    return data


def get_mean_centre(x, i):

    if i == 3:
        return x[i][0].cpu().numpy().reshape(n_channel, -1).mean(-1)
    elif i == 4:
        return x[i][0].cpu().numpy().reshape(n_channel, -1).mean(-1)
    else:
        lower_bound = 2 ** (5 - 1 - i) - int(centre_size / 2)
        upper_bound = 2 ** (5 - 1 - i) + int(centre_size / 2)
        return (
            x[i][0]
            .cpu()
            .numpy()[:, lower_bound:upper_bound, lower_bound:upper_bound]
            .reshape(n_channel, -1)
            .mean(-1)
        )

    