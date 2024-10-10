#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sheida.rahnamai@fht.org
#SBATCH --output=/group/jug/Sheida/HVAE/inference.log
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --ntasks=8
#SBATCH --job-name=3d_hvae
#SBATCH --time=50:00:00
#SBATCH --no-requeue


source $HOME/.bashrc
conda activate emseg
cd examples/3D/Pixel_Noise
python segmentation.py
