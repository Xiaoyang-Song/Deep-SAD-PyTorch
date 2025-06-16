#!/bin/bash

#SBATCH --account=sunwbgt0
#SBATCH --job-name=j16
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/sunwbgt_root/sunwbgt98/xysong/Deep-SAD-PyTorch/out/fmnist-R2-Imbalanced/16.log

mkdir log/DeepSAD
mkdir log/DeepSAD/fmnist-R2-Imbalanced-16
cd src
python main.py fmnist-R2 cifar10_LeNet ../log/DeepSAD/fmnist-R2-Imbalanced-16 ../data --ratio_known_outlier 0.01 \
                --ratio_pollution 0.1 \
        --lr 0.0001 \
        --n_epochs 150 \
        --lr_milestone 50 \
        --batch_size 128 \
        --weight_decay 0.5e-6 \
        --pretrain True \
        --normal_class "0, 1, 2, 3, 4, 5, 6, 7" \
        --known_outlier_class 1 \
        --n_known_outlier_classes 1 \
        --n_known_outlier 16 \
        --n_known_normal 1000 \
        --n_pollution 0 \
        --sampler number-pre-sampled \
        --regime Imbalanced
