#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sheida.rahnamai@fht.org
#SBATCH --output=/group/jug/Sheida/HVAE/segmentation/20/train_supervised.log
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=25GB
#SBATCH --ntasks=1
#SBATCH --job-name=segmentation
#SBATCH --time=32:00:00
#SBATCH --no-requeue


source $HOME/.bashrc
conda activate seg

# Set parameters
DIRECTORY_PATH='/group/jug/Sheida/HVAE/segmentation/20/'
SAMPLE_RATIO=20
INITIAL_MASK_SIZE=3
FINAL_MASK_SIZE=3
INITIAL_LABEL_SIZE=3
FINAL_LABEL_SIZE=3
BLOCKS_PER_LAYER=3
NUM_LATENTS=2


python Training_BetaSeg.py --directory_path $DIRECTORY_PATH

parser.add_argument("--stochastic_block_type", type=str, default="normal")
parser.add_argument("--conditional", type=bool, default=True)
parser.add_argument("--condition_type", type=str, default="mlp")
parser.add_argument("--initial_mask_size", type=int, default=1)