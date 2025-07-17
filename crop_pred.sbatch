#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sheida.rahnamai@fht.org
#SBATCH --output=/group/jug/Sheida/HVAE/segmentation/kmeans.log
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=512GB
#SBATCH --ntasks=8
#SBATCH --job-name=extac
#SBATCH --time=1-00:00:00
#SBATCH --no-requeue


source $HOME/.bashrc
conda activate epsSeg
cd examples/2D/Pixel_Noise/Convallaria/
# cd examples/3D/Pixel_Noise/
python crop_pred.py