#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sheida.rahnamai@fht.org
#SBATCH --output=/group/jug/Sheida/HVAE/segmentation/test.log
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=25GB
#SBATCH --ntasks=1
#SBATCH --job-name=segmentation
#SBATCH --time=01:00:00
#SBATCH --no-requeue


source $HOME/.bashrc
conda activate seg
python Prediction_BetaSeg.py