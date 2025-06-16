import os
import numpy as np


def read_last_auc(log_file):
    with open(log_file, 'r') as f:
        lines = f.readlines()[-4:]  # Get last 4 lines

    for line in lines:
        if 'Test AUC' in line:
            # Extract the numeric part before the '%' sign
            try:
                percent_str = line.split('Test AUC:')[1].split('%')[0].strip()
                return float(percent_str)
            except (IndexError, ValueError):
                pass  # Malformed line, skip or raise error if needed

    return None  # AUC not found


EXP_DSET = 'fmnist'
regime = 'Balanced'
# N = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
N = [8, 16, 32]

AUCs = []
TPR95 = []
TPR99 = []
for n in N:
    file_path = os.path.join('..', 'log', 'DeepSAD', f'{EXP_DSET}-{regime}-{n}', 'log.txt')
    auc_value = read_last_auc(file_path)
    # print(f"Parsed AUC: {auc_value}%")
    # print(f"Numeric AUC: {auc_value}")