#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sheida.rahnamai@fht.org
#SBATCH --output=/group/jug/Sheida/HVAE/experiments/33/train.log
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=25GB
#SBATCH --ntasks=1
#SBATCH --job-name=segmentation
#SBATCH --time=3-00:00:00
#SBATCH --no-requeue


source $HOME/.bashrc
conda activate emseg

# Set parameters
DIRECTORY_PATH='/group/jug/Sheida/HVAE/experiments/33/'
LOAD_CHECKPOINT=True
CHECKPOINT='/group/jug/Sheida/HVAE/experiments/33/model_supervised/experiments_best_vae.net'


python Training_Aitslab_bioimaging.py --directory_path $DIRECTORY_PATH --load_checkpoint $LOAD_CHECKPOINT --checkpoint $CHECKPOINT