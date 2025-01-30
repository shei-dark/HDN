import os
import numpy as np
import time
import datetime
import torch
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from boilerplate import boilerplate
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
    max_grad_norm=None,
    amp=True,
    gradient_scale=8192,
    use_wandb=True,
    initial_label_size=1,
    final_label_size=10,
    initial_mask_size=1,
    final_mask_size=10,
    step_interval=5,
    overfit_patience=20,
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
    mask_size_scheduler = boilerplate.LabelSizeScheduler(
        initial_size=initial_mask_size,
        final_size=final_mask_size,
        step_interval=step_interval,
    )
    label_size_scheduler = boilerplate.LabelSizeScheduler(
        initial_size=initial_label_size,
        final_size=final_label_size,
        step_interval=step_interval,
    )

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
            },
        )
        run.config.update(dict(epochs=max_epochs))
        wandb.run.log_code(
            ("/home/sheida.rahnamai/GIT/HDN/"),
            include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb"),
        )

    for epoch in range(max_epochs):

        print(f"Starting epoch {epoch}")

        for idx, (x, y, z) in tqdm(enumerate(train_loader), desc="Training"):
            if not use_wandb:
                if idx == 5:
                    break
            train_loader.dataset.update_patches(
                label_size_scheduler.get_label_size(epoch)
            )
            model.mask_size = mask_size_scheduler.get_label_size(epoch)
            x = x.squeeze(0)
            y = y.squeeze(0)
            x = x.to(device=device, dtype=torch.float)
            y = y.to(device=device, dtype=torch.float)

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
            ce = outputs["ce"] if outputs["ce"] is not None else torch.zeros(1)
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
                        "IP": inpainting_loss * alpha,
                        "KL": kl_loss * beta,
                        "CL": cl_loss * gamma if model.contrastive_learning else None,
                        "CE": ce,
                        "EL": entropy,
                        "Total": loss,
                    },
                    commit=True,
                )

            # Optimization step

            scaler.step(optimizer)
            scaler.update()
            model.increment_global_step()

        print("saving", model_folder + model_name + "_last_vae.net")
        torch.save(model, model_folder + model_name + "_last_vae.net")

        ### Validation step
        running_validation_loss = []

        model.eval()
        with torch.no_grad():
            for idx, (x, y, z) in tqdm(enumerate(val_loader), desc="Validation"):
                if not use_wandb:
                    if idx == 5:
                        break
                val_loader.dataset.update_patches(
                    label_size_scheduler.get_label_size(epoch)
                )
                model.mask_size = mask_size_scheduler.get_label_size(epoch)
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

                running_validation_loss.append(val_loss)
                
                if use_wandb:
                    run.log(
                        {
                            "val_IP": alpha * val_inpainting_loss,
                            "val_KL": beta * val_kl_loss,
                            "val_CE": val_ce,
                            "val_EL": val_entropy,
                            "val_CL": (
                                gamma * val_cl_loss
                                if model.contrastive_learning
                                else None
                            ),
                            "val_total": val_loss,
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
            torch.save(model.state_dict(), model_folder + model_name + "_best_weights.net")
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
        
        if patience_ > overfit_patience and model.training_mode == "supervised":
            print("Overfitting detected. Loading best model and switching to semi-supervised training...")
            model.load_state_dict(torch.load(model_folder + model_name + "_best_weights.net"))
            train_loader.dataset.switch_mode()
            val_loader.dataset.switch_mode()
            model.training_mode = "semisupervised"
            patience_ = 0

        seconds = time.time()
        secondsElapsed = float(seconds - seconds_last)
        seconds_last = seconds
        remainingEps = (max_epochs + 1) - (epoch + 1)
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

        if patience_ == 100:
            print("Early stopping")
            break