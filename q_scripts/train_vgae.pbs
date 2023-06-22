#!/bin/bash

#PBS -l select=1:ncpus=4:mem=24gb:ngpus=1:gpu_type=RTX6000
#PBS -l walltime=03:00:00
#PBS -J 1-11
 
cd $HOME
module load anaconda3/personal
source activate fyp
cd big/container/psilocybrain
 
python train_vgae.py $PBS_ARRAY_INDEX