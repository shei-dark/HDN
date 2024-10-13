import os
import glob
import random
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
from torch.cuda.amp import GradScaler

from tifffile import imread
from matplotlib import pyplot as plt
from tqdm import tqdm

from boilerplate import boilerplate
from models.lvae import LadderVAE
import lib.utils as utils
import wandb

wandb.require("core")


def train_network(
    model,
    lr,
    max_epochs,
    train_loader,
    val_loader,
    gaussian_noise_std,
    model_name,
    directory_path="./",
    batch_size=8,
    alpha=1,
    beta=1,
    gamma=1,
    nrows=4,
    max_grad_norm=None,
    amp=True,
    gradient_scale=8192,
    use_wandb=True,
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
    gaussian_noise_std: float
        standard deviation of gaussian noise (required when 'noiseModel' is None).
    model_name: String
        Name of Hierarchical DivNoising model with which to save weights.
    directory_path: String
        Path where the DivNoising weights to be saved.
    max_grad_norm: float
        Value to limit/clamp the gradients at.
    """

    model_folder = directory_path + "model/"
    device = model.device
    optimizer, scheduler = boilerplate._make_optimizer_and_scheduler(model, lr, 0.0)
    loss_train_history = []
    inpainting_loss_train_history = []
    kl_loss_train_history = []
    cl_loss_train_history = []
    loss_val_history = []

    patience_ = 0

    try:
        os.makedirs(model_folder)
    except FileExistsError:
        # directory already exists
        pass

    seconds_last = time.time()

    os.environ["WANDB_START_TIMEOUT"] = "600"

    # AMP gradscaler
    scaler = GradScaler(init_scale=gradient_scale, enabled=amp)

    if use_wandb:
        wandb.login()
        run = wandb.init(
            project=model_name,
            config={
                "learning rate": lr,
                "epochs": max_epochs,
                "batch size": batch_size,
                "inpainting loss weight": alpha,
                "KLD weight": beta,
                "contrastive learning weight": gamma,
                "lambda (cl)": model.lambda_contrastive,
                "margin": model.margin,
                "labeled ratio": model.labeled_ratio,
            },
        )
        run.config.update(dict(epochs=max_epochs))
        wandb.run.log_code(
            ("/home/sheida.rahnamai/GIT/HDN/"),
            include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb"),
        )

    global_idx = 0
    for epoch in range(max_epochs):

        print(f"Starting epoch {epoch}")
        running_training_loss = []
        running_inpainting_loss = []
        running_kl_loss = []
        running_cl_loss = []
        running_cl_pos = []
        running_cl_neg = []

        for idx, (x, y, z) in tqdm(enumerate(train_loader), desc="Training"):

            x = x.squeeze(0)
            y = y.squeeze(0)
            z = z.squeeze(0)
            x = x.to(device=device, dtype=torch.float)

            optimizer.zero_grad()

            if torch.isnan(x).any() or torch.isinf(x).any():
                print("x has nan or inf")
                continue
            outputs = boilerplate.forward_pass(
                x, y, device, model, gaussian_noise_std, amp=amp
            )

            inpainting_loss = outputs["inpainting_loss"]
            kl_loss = outputs["kl_loss"]
            cl_loss = outputs["cl_loss"]
            cl_pos = outputs["cl_pos"]
            cl_neg = outputs["cl_neg"]
            loss = alpha * inpainting_loss + beta * kl_loss + gamma * cl_loss
            with torch.autograd.set_detect_anomaly(mode=True):
                scaler.scale(loss).backward()

            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=max_grad_norm
                )

            if use_wandb:
                run.log(
                    {
                        "global_idx": global_idx,
                        "idx": idx,
                        "IP": inpainting_loss * alpha,
                        "KL": kl_loss * beta,
                        "CL": cl_loss * gamma,
                        "PPL": cl_pos,
                        "NPL": cl_neg,
                        "Total": loss,
                    },
                    commit=True,
                )
            global_idx += 1

            # Optimization step

            running_training_loss.append(loss.item())
            running_inpainting_loss.append(inpainting_loss.item())
            running_kl_loss.append(kl_loss.item())
            running_cl_loss.append(cl_loss.item())
            running_cl_pos.append(cl_pos)
            running_cl_neg.append(cl_neg)

            scaler.step(optimizer)
            scaler.update()
            model.increment_global_step()
            step = model.global_step

        ### Print training losses
        to_print = "Epoch[{}/{}] Training Loss: {:.4f} Inpainting Loss: {:.4f} KL Loss: {:.4f} CL Loss: {:.4f}"
        to_print = to_print.format(
            epoch,
            max_epochs,
            np.mean(running_training_loss),
            np.mean(running_inpainting_loss),
            np.mean(running_kl_loss),
            np.mean(running_cl_loss),
        )

        print(to_print)
        print("saving", model_folder + model_name + "_last_vae.net")
        torch.save(model, model_folder + model_name + "_last_vae.net")

        if use_wandb:
            run.log(
                {
                    "epoch": epoch,
                    "inpainting loss": np.mean(running_inpainting_loss) * alpha,
                    "kl loss": np.mean(running_kl_loss) * beta,
                    "cl loss": np.mean(running_cl_loss) * gamma,
                    "cl pos pair": torch.mean(torch.stack(running_cl_pos)).item(),
                    "cl neg pair": torch.mean(torch.stack(running_cl_neg)).item(),
                    "total loss": np.mean(running_training_loss),
                }
            )

        ### Save training losses
        loss_train_history.append(np.mean(running_training_loss))
        inpainting_loss_train_history.append(np.mean(running_inpainting_loss))
        kl_loss_train_history.append(np.mean(running_kl_loss))
        cl_loss_train_history.append(np.mean(running_cl_loss))
        np.save(model_folder + "train_loss.npy", np.array(loss_train_history))
        np.save(
            model_folder + "train_reco_loss.npy",
            np.array(inpainting_loss_train_history),
        )
        np.save(model_folder + "train_kl_loss.npy", np.array(kl_loss_train_history))
        np.save(model_folder + "train_cl_loss.npy", np.array(cl_loss_train_history))

        ### Validation step
        running_validation_loss = []
        model.eval()
        with torch.no_grad():
            for i, (x, y, z) in tqdm(enumerate(val_loader), desc="Validation"):
                x = x.squeeze(0)
                y = y.squeeze(0)
                z = z.squeeze(0)
                x = x.to(device=device, dtype=torch.float)
                val_outputs = boilerplate.forward_pass(
                    x, y, device, model, gaussian_noise_std
                )

                val_inpainting_loss = val_outputs["inpainting_loss"]
                val_kl_loss = val_outputs["kl_loss"]
                val_cl_loss = val_outputs["cl_loss"]
                val_loss = (
                    alpha * val_inpainting_loss
                    + beta * val_kl_loss
                    + gamma * val_cl_loss
                )
                running_validation_loss.append(val_loss)

        if use_wandb:
            run.log(
                {
                    "val total loss": torch.mean(
                        torch.stack(running_validation_loss)
                    ).item()
                }
            )
        model.train()

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

        print(
            "Patience:",
            patience_,
            "Validation Loss:",
            total_epoch_loss_val.item(),
            "Min validation loss:",
            np.min(loss_val_history),
        )

        seconds = time.time()
        secondsElapsed = float(seconds - seconds_last)
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
