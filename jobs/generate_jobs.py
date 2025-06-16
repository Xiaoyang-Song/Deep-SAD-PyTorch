import os
import numpy as np


# GL
ACCOUNT = 'sunwbgt0'
TIME = "2:00:00"
# Configuration
# EXP_DSET = 'fmnist'
# EXP_DSET = 'fmnist-R2'
# EXP_DSET = 'cifar10-svhn'
# EXP_DSET = 'mnist'
EXP_DSET = 'mnist-fashionmnist' 
# EXP_DSET = 'svhn' 
# EXP_DSET = 'svhn-R2'
# regime = 'Imbalanced'
regime = 'Balanced'

if EXP_DSET in ['mnist', 'fmnist', 'fmnist-R2' 'mnist-fashionmnist']:
    backbone = 'mnist_LeNet'
else:
    backbone = 'cifar10_LeNet'

N = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

CMD_ONLY = False


for n in N:
    print(f"sbatch jobs/{EXP_DSET}/{n}.sh")

if not CMD_ONLY:
    print("Generating job files...")
    # Create logging directory
    log_path = os.path.join('checkpoint', 'log', EXP_DSET)
    os.makedirs(log_path, exist_ok=True)

    for n in N:
        # Create job directory
        job_path = os.path.join('jobs', EXP_DSET)
        os.makedirs(job_path, exist_ok=True)
        # Declare job name
        filename = os.path.join('jobs', EXP_DSET, f"{n}.sh")
        # Write files
        f = open(filename, 'w')
        f.write("#!/bin/bash\n\n")
        f.write(f"#SBATCH --account={ACCOUNT}\n")
        f.write(f"#SBATCH --job-name=j{n}\n")
        f.write("#SBATCH --mail-user=xysong@umich.edu\n")
        f.write("#SBATCH --mail-type=BEGIN,END,FAIL\n")
        f.write("#SBATCH --nodes=1\n")
        f.write("#SBATCH --partition=gpu\n")
        f.write("#SBATCH --gpus=1\n")
        f.write("#SBATCH --mem-per-gpu=16GB\n")
        f.write(f"#SBATCH --time={TIME}\n")
        f.write(f"#SBATCH --output=/scratch/sunwbgt_root/sunwbgt98/xysong/Deep-SAD-PyTorch/out/{EXP_DSET}-{regime}/{n}.log\n\n")

        f.write(f"mkdir log/DeepSAD\n")
        f.write(f"mkdir log/DeepSAD/{EXP_DSET}-{regime}-{n}\n")
        f.write(f"cd src\n")
        
        f.write(
            f"""python main.py {EXP_DSET} {backbone} ../log/DeepSAD/{EXP_DSET}-{regime}-{n} ../data --ratio_known_outlier 0.01 \\
                --ratio_pollution 0.1 \\
        --lr 0.0001 \\
        --n_epochs 150 \\
        --lr_milestone 50 \\
        --batch_size 128 \\
        --weight_decay 0.5e-6 \\
        --pretrain True \\
        --normal_class "0, 1, 2, 3, 4, 5, 6, 7" \\
        --known_outlier_class 1 \\
        --n_known_outlier_classes 1 \\
        --n_known_outlier {n} \\
        --n_known_normal 1000 \\
        --n_pollution 0 \\
        --sampler number-pre-sampled \\
        --regime {regime}\n"""
        )

        f.close()
