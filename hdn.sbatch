#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sheida.rahnamai@fht.org
#SBATCH --output=output426.log
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=30GB
#SBATCH --ntasks=1
#SBATCH --job-name=hdn
#SBATCH --time=30:00:00

source $HOME/.bashrc
conda activate /home/sheida.rahnamai/miniconda3/envs/maester
# jupyter notebook --no-browser --port=8800 --ip 0.0.0.0 

cd examples/Pixel_Noise/Convallaria
python Training.py