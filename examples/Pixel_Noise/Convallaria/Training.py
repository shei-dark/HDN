import warnings
warnings.filterwarnings('ignore')
# We import all our dependencies.
import torch
import sys
sys.path.append('../../../')
sys.path.append('/home/sheida.rahnamai/GIT/HDN/')
from models.lvae import LadderVAE
from boilerplate import boilerplate
import training
import tifffile as tiff
import os

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

data_dir="/group/jug/Sheida/pancreatic beta cells/download/"
# data_dir = "/localscratch/data/"
patch_size = 64

Three_train_images = ['high_c1', 'high_c2', 'high_c3']

# Load source images
train_img_paths = [os.path.join(data_dir, img, f"{img}_source.tif") for img in Three_train_images]
train_images = {img: tiff.imread(path) for img, path in zip(Three_train_images, train_img_paths)}

# Print loaded train images paths
print("Train images loaded from paths:")
for img, path in zip(Three_train_images, train_img_paths):
   print(path)

ground_truth_images = {}
# Load ground truth images
for img in Three_train_images:
   gt_path = os.path.join(data_dir, img, f"{img}_gt.tif")
   ground_truth_images[img] = tiff.imread(gt_path)

# One_test_image = ['high_c4']

# Load test image
# test_img_path = os.path.join(data_dir, One_test_image[0], f"{One_test_image[0]}_source.tif")
# test_image = tiff.imread(test_img_path)

# Print loaded test images paths
# print("Test image loaded from path:")
# print(test_img_path)

# Load test ground truth images
# test_gt_path = os.path.join(data_dir, One_test_image[0], f"{One_test_image[0]}_gt.tif")
# test_ground_truth_image = tiff.imread(test_gt_path)


model_name = "HVAE"
# directory_path = "/group/jug/Sheida/HVAE/cl_w_bg_v8/"
directory_path = "test"
# Data-specific
gaussian_noise_std = None
noiseModel = None 
# Training-specific
batch_size=128
# batch_size=8
virtual_batch = 64
lr=3e-4
max_epochs = 500
steps_per_epoch = 2
test_batch_size=100

# Model-specific
num_latents = 3
z_dims = [32]*int(num_latents)
blocks_per_layer = 5
mask_size = 4
margin = 0
batchnorm = True
free_bits = 0.0 # if KLD is less than 1 then the loss won't be calculated
contrastive_learning = True
cl_mode = 'min max'

debug             = False #[True, False]
save_output       = True #[True, False]
use_non_stochastic = False
project           = 'HVAE'
img_shape = (64,64)

# train_loader, val_loader, test_loader, data_mean, data_std = boilerplate._make_datamanager(train_images,train_y,val_images,val_y,
train_loader, val_loader, data_mean, data_std = boilerplate._make_datamanager(train_images, ground_truth_images, batch_size)#test_image, test_ground_truth_image, batch_size)
feature_dim = 96

# val loader -> batch of patches of images with shape (1, batch_size, channel, patch_size, patch_size)
# then we have the centre label of the patch and the whole label of the patch
# the centre label is used for contrastive learning
# centre label of shape (1, batch_size)
# label of the patch of shape (1, batch_size, channel, patch_size, patch_size)

model = LadderVAE(z_dims=z_dims,blocks_per_layer=blocks_per_layer,data_mean=data_mean,data_std=data_std,noiseModel=noiseModel,
                  device=device,batchnorm=batchnorm,free_bits=free_bits,img_shape=img_shape,contrastive_learning=contrastive_learning,cl_mode=cl_mode,mask_size=mask_size, use_non_stochastic=use_non_stochastic, learn_top_prior=True, margin=margin).cuda()

model.train() # Model set in training mode

training.train_network(model=model,lr=lr,max_epochs=max_epochs,steps_per_epoch=steps_per_epoch,directory_path=directory_path,
                       train_loader=train_loader,val_loader=val_loader,
                       virtual_batch=virtual_batch,gaussian_noise_std=gaussian_noise_std,
                       model_name=model_name,val_loss_patience=100, debug=debug, save_output=save_output, project_name=project, batch_size=batch_size, cl_w = 1e-4, kl_w = 1e-1)

