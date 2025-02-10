#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sheida.rahnamai@fht.org
#SBATCH --output=/group/jug/Sheida/HVAE/experiments/26/train.log
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=25GB
#SBATCH --ntasks=8
#SBATCH --job-name=segmentation
#SBATCH --time=2-00:00:00
#SBATCH --no-requeue


source $HOME/.bashrc
conda activate emseg

# Set parameters
DIRECTORY_PATH='/group/jug/Sheida/HVAE/experiments/26/'
INITIAL_MASK_SIZE=1
FINAL_MASK_SIZE=1
INITIAL_LABEL_SIZE=1
FINAL_LABEL_SIZE=1
MODE='semisupervised'
LOAD_CHECKPOINT=True
CHECKPOINT='/group/jug/Sheida/HVAE/experiments/26/model_supervised/experiments_best_vae.net'

python Training_BetaSeg.py --initial_mask_size $INITIAL_MASK_SIZE --final_mask_size $FINAL_MASK_SIZE --initial_label_size $INITIAL_LABEL_SIZE --final_label_size $FINAL_LABEL_SIZE --directory_path $DIRECTORY_PATH --mode $MODE --load_checkpoint $LOAD_CHECKPOINT --checkpoint $CHECKPOINT