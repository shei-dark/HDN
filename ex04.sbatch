#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sheida.rahnamai@fht.org
#SBATCH --output=/group/jug/Sheida/HVAE/segmentation/04/train_supervised.log
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=25GB
#SBATCH --ntasks=1
#SBATCH --job-name=segmentation
#SBATCH --time=32:00:00
#SBATCH --no-requeue


source $HOME/.bashrc
conda activate seg

# Set parameters
DIRECTORY_PATH='/group/jug/Sheida/HVAE/segmentation/04/'
STOCHASTIC_BLOCK_TYPE='normal'


python Training_BetaSeg.py --directory_path $DIRECTORY_PATH --stochastic_block_type $STOCHASTIC_BLOCK_TYPE