#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sheida.rahnamai@fht.org
#SBATCH --output=/group/jug/Sheida/HVAE/experiments/21/train.log
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=25GB
#SBATCH --ntasks=8
#SBATCH --job-name=segmentation
#SBATCH --time=3-00:00:00
#SBATCH --no-requeue


source $HOME/.bashrc
conda activate emseg

# Set parameters
DIRECTORY_PATH='/group/jug/Sheida/HVAE/experiments/21/'
OVERFIT_PATIENCE=20
CONTRASTIVE_LEARNING=True
STOCHASTIC_BLOCK_TYPE='mixture'
CONDITIONAL=True
CONDITION_TYPE='mlp'
SAMPLE_RATIO=5

python Training_BetaSeg.py --directory_path $DIRECTORY_PATH --overfit_patience $OVERFIT_PATIENCE --contrastive_learning $CONTRASTIVE_LEARNING --stochastic_block_type $STOCHASTIC_BLOCK_TYPE --conditional $CONDITIONAL --condition_type $CONDITION_TYPE --sample_ratio $SAMPLE_RATIO