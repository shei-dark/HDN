#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sheida.rahnamai@fht.org
#SBATCH --output=/group/jug/Sheida/HVAE/segmentation/31/test.log
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=512GB
#SBATCH --ntasks=1
#SBATCH --job-name=segmentation
#SBATCH --time=5-00:00:00
#SBATCH --no-requeue


source $HOME/.bashrc
conda activate epsSeg
python Training_Semisupervised_BetaSeg.py