#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sheida.rahnamai@fht.org
#SBATCH --output=/group/jug/Sheida/HVAE/segmentation/prediction_19_supervised.log
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=5GB
#SBATCH --ntasks=1
#SBATCH --job-name=segmentation
#SBATCH --time=1-00:00:00
#SBATCH --no-requeue


source $HOME/.bashrc
conda activate seg
python Prediction_MarinoLiver.py