#!/bin/bash

#SBATCH --account=sunwbgt0
#SBATCH --job-name=SAD-M
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/sunwbgt_root/sunwbgt98/xysong/Deep-SAD-PyTorch/out/out-mnist.log


mkdir log/DeepSAD
mkdir log/DeepSAD/mnist

# change to source directory
cd src

# run experiment
python main.py mnist mnist_LeNet ../log/DeepSAD/mnist ../data --ratio_known_outlier 0.01 --ratio_pollution 0.1 \
    --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 128 --weight_decay 0.5e-6 --pretrain True \
    --ae_lr 0.0001 --ae_n_epochs 150 --ae_batch_size 128 --ae_weight_decay 0.5e-3 \
    --normal_class "0" \
    --known_outlier_class 1 \
    --n_known_outlier_classes 1 \
    --n_known_outlier 1000 \
    --n_known_normal 1000 \
    --n_pollution 0 \
    --sampler original