#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sheida.rahnamai@fht.org
#SBATCH --output="/group/jug/Sheida/HVAE/WACV/epsSeg/train.log"
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --ntasks=1
#SBATCH --job-name=WACV
#SBATCH --time=2-00:00:00
#SBATCH --no-requeue


source $HOME/.bashrc
conda activate epsSeg
python Training_Semisupervised_BetaSeg.py