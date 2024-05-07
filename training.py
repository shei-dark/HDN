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


def train_network(model, lr, max_epochs,steps_per_epoch,train_loader, val_loader, test_loader, 
                  virtual_batch, gaussian_noise_std, model_name, 
                  test_log_every=1000, directory_path="./",
                  val_loss_patience=100, nrows=4, max_grad_norm=None, debug=False, project_name="", save_output=True, batch_size=8, cl_w = 1e-7, kl_w = 1):
    
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
    if debug == False:
        use_wandb = True
    else:
        use_wandb = False

    experiment = wandb.init(project = project_name,
                                resume = 'allow',
                                anonymous = 'must',
                                mode = 'online',
                                reinit = True,
                                save_code = True)

    experiment.config.update(dict(epochs=max_epochs,batch_size=batch_size,))

    wandb.run.log_code(("."), include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb"))

    model_folder = directory_path+"model/"
    img_folder = directory_path+"imgs/"
    device = model.device
    optimizer, scheduler = boilerplate._make_optimizer_and_scheduler(model,lr,0.0)
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
    
    try:
        os.makedirs(img_folder)
    except FileExistsError:
    # directory already exists
        pass
    
    seconds_last = time.time()
    
    while step_counter / steps_per_epoch < max_epochs:
        epoch = epoch+1
        running_training_loss = []
        running_reconstruction_loss = []
        running_kl_loss = []
        running_cl_loss = []
        # if epoch % 10 == 0 and epoch <=50:
        #     cl_w *= 10
        #     kl_w /= 10
        for batch_idx, (x, y) in enumerate(train_loader):
            step_counter=batch_idx
            x = x.unsqueeze(1) # Remove for RGB
            x = x.to(device=device, dtype=torch.float)
            # step = model.global_step
            
            # if(test_log_every > 0):
            #     if step % test_log_every == 0:
                
            #         print("Testing the model at " "step {}". format(step))

            #         with torch.no_grad():
            #             boilerplate._test(epoch, img_folder, device, model,
            #                               test_loader, gaussian_noise_std,
            #                               model.data_std, nrows)
            #             model.train()
            model.train()
            optimizer.zero_grad()
        
        
            ### Make smaller batches
            virtual_batches = torch.split(x,virtual_batch,0)
            virtual_batches_y = torch.split(y,virtual_batch,0)
            # for batch in virtual_batches:
            for indx in range(len(virtual_batches)):
                batch = virtual_batches[indx]
                batch_label = virtual_batches_y[indx]
                outputs = boilerplate.forward_pass(batch, batch_label, device, model, 
                                                                gaussian_noise_std)

                recons_loss = outputs['recons_loss']
                kl_loss = outputs['kl_loss']
                cl_loss = outputs['cl_loss']
                if model.contrastive_learning and not model.use_non_stochastic:
                    loss = recons_loss + kl_w * kl_loss + cl_w * cl_loss
                elif model.use_non_stochastic:
                    loss = recons_loss + cl_w * cl_loss
                else:
                    loss = recons_loss + kl_w * kl_loss 
                loss.backward()

                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                # Optimization step

                running_training_loss.append(loss.item())
                running_reconstruction_loss.append(recons_loss.item())
                # running_kl_loss.append(kl_loss.item())
                running_cl_loss.append(cl_loss.item())
            
            optimizer.step()
            
            # model.increment_global_step()
            
            # first_step = False
            # if step_counter % steps_per_epoch == steps_per_epoch-1:
            if True:  
                ## Print training losses
                to_print = "Epoch[{}/{}] Training Loss: {:.3f} Reconstruction Loss: {:.3f} CL Loss: {:.3f}"# KL Loss: {:.3f} CL Loss: {:.3f}"
                to_print = to_print.format(epoch,
                                          max_epochs, 
                                          np.mean(running_training_loss),
                                          np.mean(running_reconstruction_loss),
                                        #   np.mean(running_kl_loss),
                                          np.mean(running_cl_loss))
                experiment.log({
                            'epoch': epoch,
                            'max_epochs': max_epochs,
                            'recons_loss': np.mean(running_reconstruction_loss),
                            # 'kl_loss': np.mean(running_kl_loss),
                            'cl_loss': np.mean(running_cl_loss),
                            'loss': np.mean(running_training_loss)
                        })
                print(to_print)
                print('saving',model_folder+model_name+"_last_vae.net")
                torch.save(model, model_folder+model_name+"_last_vae.net")

                ### Save training losses 
                loss_train_history.append(np.mean(running_training_loss))
                reconstruction_loss_train_history.append(np.mean(running_reconstruction_loss))
                # kl_loss_train_history.append(np.mean(running_kl_loss))
                cl_loss_train_history.append(np.mean(running_cl_loss))
                np.save(model_folder+"train_loss.npy", np.array(loss_train_history))
                np.save(model_folder+"train_reco_loss.npy", np.array(reconstruction_loss_train_history))
                # np.save(model_folder+"train_kl_loss.npy", np.array(kl_loss_train_history))
                np.save(model_folder+"train_cl_loss.npy", np.array(cl_loss_train_history))
        
        
                ### Validation step
                running_validation_loss = []
                model.eval()
                with torch.no_grad():
                    for i, (x, y) in enumerate(val_loader):
                        x = x.unsqueeze(1) # Remove for RGB
                        x = x.to(device=device, dtype=torch.float)
                        val_outputs = boilerplate.forward_pass(x, y, device, model, gaussian_noise_std)

                        val_recons_loss = val_outputs['recons_loss']
                        # val_kl_loss = val_outputs['kl_loss']
                        val_cl_loss = val_outputs['cl_loss']
                        # val_loss = val_recons_loss + kl_w * val_kl_loss + cl_w * val_cl_loss
                        val_loss = val_recons_loss + cl_w * val_cl_loss
                        running_validation_loss.append(val_loss)
                model.train()

                total_epoch_loss_val = torch.mean(torch.stack(running_validation_loss))
                scheduler.step(total_epoch_loss_val)

                ### Save validation losses      
                loss_val_history.append(total_epoch_loss_val.item())
                np.save(model_folder+"val_loss.npy", np.array(loss_val_history))

                if total_epoch_loss_val.item() < 1e-6 + np.min(loss_val_history):
                    patience_ = 0
                    print('saving',model_folder+model_name+"_best_vae.net")
                    torch.save(model, model_folder+model_name+"_best_vae.net")
                    
                else:
                    patience_ +=1

                if epoch % 10 == 0:
                    torch.save(model.state_dict(), model_folder+"scheduled_weight_"+str(epoch)+".net")

                print("Patience:", patience_,
                      "Validation Loss:", total_epoch_loss_val.item(),
                      "Min validation loss:", np.min(loss_val_history))

                seconds=time.time()
                secondsElapsed=np.float64(seconds-seconds_last)
                seconds_last=seconds
                remainingEps=(max_epochs+1)-(epoch+1)
                estRemainSeconds=(secondsElapsed)*(remainingEps)
                estRemainSecondsInt=int(secondsElapsed)*(remainingEps)
                print('Time for epoch: '+ str(int(secondsElapsed))+ 'seconds')

                print('Est remaining time: '+
                      str(datetime.timedelta(seconds= estRemainSecondsInt)) +
                      ' or ' +
                      str(estRemainSecondsInt)+ 
                      ' seconds')

                print("----------------------------------------", flush=True)

                if patience_ == val_loss_patience:
#                     print("Employing early stopping, validation loss did not improve for 100 epochs !"
                    return
                
                break