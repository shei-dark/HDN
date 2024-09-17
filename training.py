import numpy as np
import time
import datetime
import torch
import os
from boilerplate import boilerplate
import wandb
from tqdm import tqdm
from lib.logging import log_all_plots
from lib.utils import BetaScheduler, change_prior_variance

wandb.require("core")


def train_network(
    model,
    lr,
    max_epochs,
    steps_per_epoch,
    train_loader,
    val_loader,
    test_set,
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
    cl_w=1,
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

    model_folder = directory_path + "model/"
    device = model.device
    optimizer, scheduler = boilerplate._make_optimizer_and_scheduler(model, lr, 0.0)
    beta = model.beta
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
    try:
        os.makedirs(model_folder)
    except FileExistsError:
        # directory already exists
        pass


    seconds_last = time.time()
    os.environ["WANDB_START_TIMEOUT"] = "600"
    # beta_scheduler = BetaScheduler(
    # initial_value= beta,            # The initial value of the hyperparameter
    # end_value= 0.3,                # The value after the decay period
    # total_epochs=100,             # Total epochs over which the decay happens
    # start_stable_epochs=10,       # The first 10 epochs where the value remains stable
    # end_stable_epochs=10          # The last 10 epochs where the value remains stable
    # )

    wandb.login()
    if debug == False:
        use_wandb = True
    else:
        use_wandb = False

    if use_wandb:
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
        wandb.run.log_code(("/home/sheida.rahnamai/GIT/HDN/"), include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb"))
    try:
        while epoch < max_epochs:
            epoch = epoch + 1
            running_training_loss = []
            running_reconstruction_loss = []
            running_kl_loss = []
            running_cl_loss = []

            # if epoch % 10 == 0:
            #     for top_down_layer in model.top_down_layers:
            #         if top_down_layer.is_top_layer:
            #             top_down_layer.top_prior_params.data[:, 32:, :, :] = change_prior_variance(epoch)
     

            for batch_idx, (x, y, _) in enumerate(train_loader):
                step_counter = batch_idx
                x = x.squeeze(0).to(device=device, dtype=torch.float)
                model.mode_pred = False
                model.train()
                optimizer.zero_grad()

                outputs = boilerplate.forward_pass(x, y, device, model, gaussian_noise_std)

                ppl = outputs['ppl']
                npl = outputs["npl"]
                npl_sum = outputs["npl_sum"]
                alphas = outputs["alphas"]

                recons_loss = outputs["recons_loss"]
                kl_loss = outputs["kl_loss"]
                cl_loss = outputs["cl_loss"]
                if model.contrastive_learning and not model.use_non_stochastic:
                    loss = recons_loss + kl_w * kl_loss + cl_w * cl_loss
                elif model.use_non_stochastic:
                    loss = recons_loss + cl_w * cl_loss
                else:
                    loss = recons_loss + kl_w * kl_loss
                if torch.isnan(loss).any():
                    continue
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
            # if step_counter % steps_per_epoch == steps_per_epoch - 1:
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
                            "kl_loss": np.mean(running_kl_loss) * kl_w,
                            "cl_loss": np.mean(running_cl_loss) * cl_w,
                            "loss": np.mean(running_training_loss),
                            "kl_weight": kl_w,
                            "cl_weight": cl_w,
                            "beta": model.beta,
                            # "beta": beta,
                            "negative pair loss": npl_sum,
                            "positive pair loss": ppl,
                            "margin": model.margin,
                        }
                    )
                    for key, value in npl.items():
                        run.log({f"{key}": value})
                        run.log({f"alpha_{key}": alphas[key]})
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
                np.save(model_folder + "train_kl_loss.npy", np.array(kl_loss_train_history))
                np.save(model_folder + "train_cl_loss.npy", np.array(cl_loss_train_history))

                ### Validation step
                running_validation_loss = []
                val_cl = []
                val_kl = []
                val_inpainting = []
                model.eval()
                with torch.no_grad():
                    for i, (x, y, _) in enumerate(val_loader):
                        x = x.squeeze(0)  # Remove for RGB
                        x = x.to(device=device, dtype=torch.float)
                        val_outputs = boilerplate.forward_pass(
                            x, y, device, model, gaussian_noise_std
                        )

                        val_recons_loss = val_outputs["recons_loss"]
                        val_kl_loss = val_outputs["kl_loss"]
                        val_cl_loss = val_outputs["cl_loss"]
                        val_loss = val_recons_loss + kl_w * val_kl_loss + cl_w * val_cl_loss

                        # val_loss = val_recons_loss + cl_w * val_cl_loss
                        running_validation_loss.append(val_loss)
                        val_cl.append(val_cl_loss * cl_w)
                        val_kl.append(val_kl_loss * kl_w)
                        val_inpainting.append(val_recons_loss)
                    if use_wandb:
                        run.log(
                            {
                                "val_inpainting_loss": torch.mean(
                                    torch.stack(val_inpainting)
                                ),
                                "val_kl_loss": torch.mean(torch.stack(val_kl)),
                                "val_cl_loss": torch.mean(torch.stack(val_cl)),
                                "val_loss": torch.mean(
                                    torch.stack(running_validation_loss)
                                ),
                            }
                        )

                    if test_set is not None:
                        log_all_plots(wandb, test_set, model)

                total_epoch_loss_val = torch.mean(torch.stack(running_validation_loss))
                scheduler.step(total_epoch_loss_val)
                # beta_scheduler.step()
                # model.beta = beta_scheduler.get_value()
                # kl_w, cl_w = update_loss_weights(kl_w, cl_w, np.mean(running_kl_loss), np.mean(running_cl_loss), np.mean(running_reconstruction_loss))

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
                    print("Early stopping")
                    return
    except Exception as e:
        print(f"An error occurred: {e}")
        raise
            # break
    if use_wandb:
        run.finish()
