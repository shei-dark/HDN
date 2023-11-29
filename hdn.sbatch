#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sheida.rahnamai@fht.org
#SBATCH --output=hdn.log
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=4GB
#SBATCH --ntasks=1
#SBATCH --job-name=hdn
#SBATCH --time=24:00:00

conda activate maester
cd examples/Pixel_Noise/Convallaria
python training.py

