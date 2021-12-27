#!/bin/sh
#SBATCH -p gpu-v100-32gb
#SBATCH --job-name=py_test
#SBATCH -o py_out%j.out
#SBATCH -e py_err%j.err
#SBATCH -n 12
#SBATCH -N 1
#SBATCH --gres=gpu:2

# creates a python virtual environment
module load python3/anaconda/2019.07
module load cuda/10.0
module load gcc/7.3.0
source activate cyclegan

cd /home/lanf/work/N2D/KPN-Single-Image/

model_path=$1
sample_path=$2
lamda=$3
loss_det=$4
batch_size=$5
#python -m visdom.server
# run python script
python train.py \
--multi_gpu True \
--save_path $model_path \
--sample_path $sample_path \
--save_mode 'epoch' \
--save_by_epoch 20 \
--save_by_iter 10000 \
--lr_g 0.0002 \
--b1 0.5 \
--b2 0.999 \
--weight_decay 0.0 \
--train_batch_size $batch_size \
--epochs 201 \
--lr_decrease_epoch 50 \
--num_workers 16 \
--mu 0 \
--sigma 30 \
--mixture_width 3 \
--lamda $lamda \
--loss_det $loss_det \
# exit the virtual environment
conda deactivate
