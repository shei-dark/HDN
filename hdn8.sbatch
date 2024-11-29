#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sheida.rahnamai@fht.org
#SBATCH --output=/group/jug/Sheida/HVAE/TAC/ex_8.log
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --ntasks=8
#SBATCH --job-name=extac
#SBATCH --time=3-00:00:00
#SBATCH --no-requeue


source $HOME/.bashrc
conda activate emseg
cd examples/2D/Pixel_Noise/Convallaria/
# cd examples/3D/Pixel_Noise/
python Training8.py