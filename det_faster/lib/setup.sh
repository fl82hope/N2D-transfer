#!/bin/sh
#SBATCH -p gpu-v100-32gb
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
source activate faster-rcnn

cd /home/lanf/work/N2D+trans/KPN-Single-Image/det_faster/lib/

# run python script
python setup.py build develop
# exit the virtual environment
conda deactivate
