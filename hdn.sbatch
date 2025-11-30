#!/bin/bash
#SBATCH --job-name=segmentation
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sheida.rahnamai@fht.org
#SBATCH --partition=dgx
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:35gb:1
#SBATCH --mem-per-cpu=32G
#SBATCH --output="/group/jug/Sheida/HVAE/segmentation/06/train+.log"


source $HOME/.bashrc
conda activate epsSeg
python Training_Semisupervised_BetaSeg.py