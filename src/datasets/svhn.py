from torch.utils.data import Subset
from PIL import Image
from torchvision.datasets import CIFAR10, SVHN
from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import create_semisupervised_setting, create_semisupervised_setting_number, get_target_label_idx
import os
from collections import Counter

import torch
import torchvision.transforms as transforms
import random
import numpy as np


class SVHN_Dataset_Customized(TorchvisionDataset):

    def __init__(self, root: str, normal_class: int = 0, known_outlier_class: int = 1, 
                 n_known_outlier_classes=None, ratio_known_normal=None, ratio_known_outlier=None, ratio_pollution=None,
                 n_known_normal=None, n_known_outlier=None, n_pollution=None,
                 sampler: str="original", regime=None):
        super().__init__(root)

        # Define normal and outlier classes
        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 10))

        if type(normal_class) == int:
            self.outlier_classes.remove(normal_class)
        else:
            self.outlier_classes = list(set(self.outlier_classes) - set(normal_class))
        print(f"Outlier classes: {self.outlier_classes}")

          # MNIST preprocessing: feature scaling to [0, 1]
        transform = transforms.ToTensor()
        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        # Get train set
        train_set = MySVHN(root=self.root, split='train', download=True, transform=transform, target_transform=target_transform)
        print(f"Original training data size: ", len(train_set), train_set.data.shape)

        if n_known_outlier_classes == 0:
            self.known_outlier_classes = ()
        elif n_known_outlier_classes == 1:
            self.known_outlier_classes = tuple([known_outlier_class])
        else:
            # self.known_outlier_classes = tuple(random.sample(self.outlier_classes, n_known_outlier_classes))
            self.known_outlier_classes = self.outlier_classes


        # Create semi-supervised setting
        if sampler == "number-pre-sampled":
            assert regime is not None and n_known_outlier is not None, \
                "If sampler is 'number-pre-sampled', regime and n_outlier must be provided."
            print("Using number-pre-sampled sampler")
            train_idx_normal = get_target_label_idx(train_set.labels, self.normal_classes)
            # InD
            InD_train_set = train_set.data[train_idx_normal].transpose((0, 2, 3, 1))
            InD_train_targets = train_set.labels[train_idx_normal]
            print(f"Train set shape: {InD_train_set.shape}")

            # OOD from pre-sampled data
            OoD_path = os.path.join("..", "..", "Out-of-Distribution-GANs", "checkpoint", "OOD-Sample", "SVHN", f"OOD-{regime}-{n_known_outlier}.pt")
            OoD_data, OoD_labels = torch.load(OoD_path)
            OoD_train_set = np.array(OoD_data.squeeze())
            OoD_train_set = (OoD_train_set.transpose((0, 2, 3, 1)) * 255).astype(np.uint8)

            train_set = torch.tensor(np.concatenate((InD_train_set, OoD_train_set), axis=0))
            train_targets = np.concatenate((InD_train_targets, OoD_labels.numpy()), axis=0)
            semi_targets = np.concatenate((np.ones(len(InD_train_targets)), -np.ones(len(OoD_labels))), axis=0)
            print(f"Train set size: {len(train_set)}")
            print(f"Train targets size: {len(train_targets)}")
            print(f"Train semi-targets size: {len(semi_targets)}")
            print(Counter(train_targets))

            self.train_set = MySVHN(root=self.root, split='train', download=True, transform=transform, target_transform=target_transform)
            self.train_set.data = train_set
            self.train_set.targets = train_targets
            self.train_set.semi_targets = semi_targets

        # Get test set
        self.test_set = MySVHN(root=self.root, split='test', download=True, transform=transform, target_transform=target_transform)
        self.test_set.data = self.test_set.data.transpose((0, 2, 3, 1))  # SVHN data is in (N, C, H, W) format


class MySVHN(SVHN):
    """
    Torchvision CIFAR10 class with additional targets for the semi-supervised setting and patch of __getitem__ method
    to also return the semi-supervised target as well as the index of a data sample.
    """

    def __init__(self, *args, **kwargs):
        super(MySVHN, self).__init__(*args, **kwargs)

        self.semi_targets = torch.zeros(len(self.labels), dtype=torch.int64)
        # self.semi_targets = None

    def __getitem__(self, index):
        """Override the original method of the CIFAR10 class.
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, semi_target, index)
        """
        img, target, semi_target = self.data[index], self.labels[index], int(self.semi_targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if type(img) != np.ndarray:
            img = img.numpy()
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, semi_target, index
