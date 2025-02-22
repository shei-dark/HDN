#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sheida.rahnamai@fht.org
#SBATCH --output=/group/jug/Sheida/HVAE/segmentation/09/train_semisupervised.log
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=26GB
#SBATCH --ntasks=1
#SBATCH --job-name=segmentation
#SBATCH --time=5:00:00
#SBATCH --no-requeue


source $HOME/.bashrc
conda activate emseg

# Set parameters
DIRECTORY_PATH='/group/jug/Sheida/HVAE/segmentation/09/'
MODE='semisupervised'
LOAD_CHECKPOINT=True
CHECKPOINT='/group/jug/Sheida/HVAE/segmentation/09/model_supervised/segmentation_best_vae.net'
SAMPLE_RATIO=1

python Training_BetaSeg.py --directory_path $DIRECTORY_PATH --mode $MODE --load_checkpoint $LOAD_CHECKPOINT --checkpoint $CHECKPOINT --sample_ratio $SAMPLE_RATIO