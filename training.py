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

# import optuna

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
    trial=None,
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
    cross_entropy_loss_train_history = []
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
        running_ce_loss = []
        running_cl_loss = []
        running_entropy_loss = []

        # Parameters
        initial_size = 6
        final_size = 1
        step_interval = 5  # Change every 5 steps

        for idx, (x, y, z) in tqdm(enumerate(train_loader), desc="Training"):

            x = x.squeeze(0)
            y = y.squeeze(0)
            x = x.to(device=device, dtype=torch.float)
            y = y.to(device=device, dtype=torch.float)

            optimizer.zero_grad()

            if torch.isnan(x).any() or torch.isinf(x).any():
                print("x has nan or inf")
                continue
            outputs = boilerplate.forward_pass(
                x, y, device, model, gaussian_noise_std, amp=amp, epoch=epoch
            )

            inpainting_loss = outputs["inpainting_loss"]
            kl_loss = outputs["kl_loss"]
            repulsive = outputs["repulsive"]
            cl_loss = outputs["cl_loss"]
            ce = outputs["ce"]
            entropy = outputs["entropy"]

            loss = alpha * inpainting_loss + beta * kl_loss + ce + entropy
            if model.contrastive_learning:
                loss += gamma * cl_loss

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
                        "Repulsive": repulsive,
                        "CL": cl_loss * gamma if model.contrastive_learning else None,
                        "Total": loss,
                        "CE": ce,
                        "EL": entropy,
                    },
                    commit=True,
                )
            global_idx += 1

            # Optimization step

            running_training_loss.append(loss)
            running_inpainting_loss.append(inpainting_loss)
            running_kl_loss.append(kl_loss)
            running_ce_loss.append(ce)
            running_entropy_loss.append(entropy)
            if model.contrastive_learning:
                running_cl_loss.append(cl_loss)

            scaler.step(optimizer)
            scaler.update()
            model.increment_global_step()
            step = model.global_step

        print("saving", model_folder + model_name + "_last_vae.net")
        torch.save(model, model_folder + model_name + "_last_vae.net")

        if use_wandb:
            run.log(
                {
                    "epoch": epoch,
                    "inpainting loss": torch.mean(torch.stack(running_inpainting_loss))
                    * alpha,
                    "kl loss": torch.mean(torch.stack(running_kl_loss)) * beta,
                    "ce loss": torch.mean(torch.stack(running_ce_loss)),
                    # "entropy loss": torch.mean(torch.stack(running_entropy_loss)),
                    "total loss": torch.mean(torch.stack(running_training_loss)),
                }
            )
            if model.contrastive_learning:
                run.log(
                    {
                        "cl loss": torch.mean(torch.stack(running_cl_loss)) * gamma,
                    }
                )

        ### Validation step
        running_validation_loss = []
        running_val_inpainting_loss = []
        running_val_kl_loss = []
        running_val_ce_loss = []
        running_val_cl_loss = []
        running_val_entropy_loss = []

        model.eval()
        with torch.no_grad():
            for i, (x, y, z) in tqdm(enumerate(val_loader), desc="Validation"):

                x = x.squeeze(0)
                y = y.squeeze(0)
                z = z.squeeze(0)
                x = x.to(device=device, dtype=torch.float)
                y = y.to(device=device, dtype=torch.float)
                val_outputs = boilerplate.forward_pass(
                    x, y, device, model, gaussian_noise_std
                )

                val_inpainting_loss = val_outputs["inpainting_loss"]
                val_kl_loss = val_outputs["kl_loss"]
                val_ce = val_outputs["ce"]
                val_entropy = val_outputs["entropy"]
                val_cl_loss = (
                    val_outputs["cl_loss"] if model.contrastive_learning else 0
                )
                val_loss = (
                    alpha * val_inpainting_loss
                    + beta * val_kl_loss
                    + val_ce
                    + val_entropy
                )
                if model.contrastive_learning:
                    val_loss += gamma * val_cl_loss
                    running_val_cl_loss.append(gamma * val_cl_loss)

                running_validation_loss.append(val_loss)
                running_val_inpainting_loss.append(alpha * val_inpainting_loss)
                running_val_kl_loss.append(beta * val_kl_loss)
                running_val_ce_loss.append(val_ce)
                running_val_entropy_loss.append(val_entropy)

        if use_wandb:
            run.log(
                {
                    "val total loss": torch.mean(
                        torch.stack(running_validation_loss)
                    ).item(),
                    "val inpainting loss": torch.mean(
                        torch.stack(running_val_inpainting_loss)
                    ).item(),
                    "val kl loss": torch.mean(torch.stack(running_val_kl_loss)).item(),
                    "val ce": torch.mean(torch.stack(running_val_ce_loss)).item(),
                    # "val entropy": torch.mean(torch.stack(running_val_entropy_loss)).item(),
                    "val cl loss": (
                        torch.mean(torch.stack(running_val_cl_loss)).item()
                        if model.contrastive_learning
                        else 0
                    ),
                }
            )
        # beta /= 5
        gamma /= 5
        # if trial is not None:
        #     trial.report(torch.mean(torch.stack(running_val_cl_loss)).item(), epoch)
        #     if trial.should_prune():
        #         raise optuna.exceptions.TrialPruned()

        model.train()

        total_epoch_loss_val = torch.mean(torch.stack(running_validation_loss))
        scheduler.step(total_epoch_loss_val)

        # TODO increasing/decreasing the label size
        # label_size = boilerplate.label_size_scheduler(
        #     initial_size=initial_size,
        #     final_size=final_size,
        #     step_interval=step_interval,
        #     current_step=epoch,
        # )
        # train_loader.dataset.update_patches(label_size)
        # val_loader.dataset.update_patches(label_size)

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
    # return torch.mean(torch.stack(running_val_cl_loss)).item()


def train_unet(unet, train_loader, val_loader, epochs=50, lr=3e-4, device="cuda"):
    unet.to(device)
    optimizer = optim.Adam(unet.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        unet.train()
        train_loss = 0
        for patches, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            patches, labels = patches.to(device), labels.to(device)
            optimizer.zero_grad()
            center_preds = unet(patches)
            loss = criterion(center_preds, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss = 0
        unet.eval()
        with torch.no_grad():
            for patches, labels in tqdm(val_loader, desc="Validating"):
                patches, labels = patches.to(device), labels.to(device)
                center_preds = unet(patches)
                val_loss += criterion(center_preds, labels).item()

        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )
