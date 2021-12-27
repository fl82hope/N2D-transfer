#!/bin/sh
#SBATCH -p gpu-v100-16gb
#SBATCH --job-name=py_test
#SBATCH -o py_out%j.out
#SBATCH -e py_err%j.err
#SBATCH -n 12
#SBATCH -N 1
#SBATCH --gres=gpu:1

# creates a python virtual environment
module load python3/anaconda/2019.07
module load cuda/10.0
module load gcc/7.3.0
source activate cyclegan

cd /home/lanf/work/N2D/KPN-Single-Image/

#python -m visdom.server
# run python script
cont=$1
python transfer.py \
--content $cont \
--style ./datasets/Style \
--output ./datasets/Cars_aug
# exit the virtual environment
conda deactivate
