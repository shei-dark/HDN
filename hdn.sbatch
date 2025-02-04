#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sheida.rahnamai@fht.org
#SBATCH --output=/group/jug/Sheida/HVAE/experiments/17/train.log
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
# cd examples/2D/Pixel_Noise/Convallaria/
# cd examples/3D/Pixel_Noise/
python Training_BetaSeg.py