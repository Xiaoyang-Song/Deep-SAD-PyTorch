from torch.utils.data import Subset
from PIL import Image
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, SVHN
from base.torchvision_dataset import TorchvisionDataset
from torchvision.datasets.vision import VisionDataset
from .preprocessing import create_semisupervised_setting, create_semisupervised_setting_number, get_target_label_idx
from collections import Counter
import torch
import torchvision.transforms as transforms
import random
import numpy as np
import os


class BTN_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class: int = 0, known_outlier_class: int = 1, 
                 n_known_outlier_classes=None, ratio_known_normal=None, ratio_known_outlier=None, ratio_pollution=None,
                 n_known_normal=None, n_known_outlier=None, n_pollution=None,
                 sampler: str="original", regime=None, ind=None, ood=None):
        super().__init__(root)

        # Define normal and outlier classes
        self.n_classes = 2  # 0: normal, 1: outlier
        self.ind = ind
        self.ood = ood

        # MNIST preprocessing: feature scaling to [0, 1]
        transform = transforms.ToTensor()
        target_transform = None

        if self.ind == "mnist":
            c=1
            ind_train = MNIST(root=self.root, train=True, download=True, transform=transform, target_transform=target_transform)
            ind_test = MNIST(root=self.root, train=False, download=True, transform=transform, target_transform=target_transform)
            train_set = ind_train.train_data
            ind_test_set = ind_test.test_data
        
        elif self.ind == 'cifar10':
            c=3
            ind_train = CIFAR10(root=self.root, train=True, download=True, transform=transform, target_transform=target_transform)
            ind_test = CIFAR10(root=self.root, train=False, download=True, transform=transform, target_transform=target_transform)
            ind_train_set = ind_train.data
            ind_test_set = ind_test.data
        print(f"InD Train set shape: {ind_train.data.shape}, InD Test set shape: {ind_test.data.shape}")

        # OOD
        assert ood in ['mnist', 'fashionmnist', 'svhn'], f"Unknown dataset: {ood}"
        if ood == 'svhn':
            # ood training
            OoD_path = os.path.join("..", "..", "Out-of-Distribution-GANs", "checkpoint", "OOD-Sample", "CIFAR10-SVHN", f"OOD-{regime}-{n_known_outlier}.pt")
            OoD_data, OoD_labels = torch.load(OoD_path)
            OoD_train_set = np.array(OoD_data.squeeze())
            OoD_train_set = (OoD_train_set.transpose((0, 2, 3, 1)) * 255).astype(np.uint8)  # loaded SVHN data is in (N, C, H, W) format
            print(OoD_train_set.shape)
            # print(OoD_train_set[0])
            # ood testing
            ood_test = SVHN(root=self.root, split='test', download=True, transform=transform, target_transform=target_transform)
            ood_test.data = ood_test.data.transpose((0, 2, 3, 1))  # SVHN data is in (N, C, H, W) format
            # print(f"OoD Test set shape: {ood_test.data.shape}")
            ood_test_set = ood_test.data
        elif ood == 'fashionmnist':
            # ood training
            OoD_path = os.path.join("..", "..", "Out-of-Distribution-GANs", "checkpoint", "OOD-Sample", "FashionMNIST", f"OOD-{regime}-{n_known_outlier}.pt")
            OoD_data, OoD_labels = torch.load(OoD_path)
            OoD_train_set = np.array(OoD_data.squeeze())
            # ood testing
            ood_test = FashionMNIST(root=self.root, train=False, download=True, transform=transform, target_transform=target_transform)
            ood_test_set = ood_test.test_data
        print(f"OoD Test set shape: {ood_test.data.shape}")

        # Create semi-supervised setting (only for setting of SEE-OOD paper; not for original DeepSAD paper)
        assert sampler == "number-pre-sampled"
        if sampler == "number-pre-sampled":
            assert regime is not None and n_known_outlier is not None, \
                "If sampler is 'number-pre-sampled', regime and n_outlier must be provided."
            print("Using number-pre-sampled sampler")

            #InD
            InD_train_set = ind_train.data
            InD_train_targets = ind_train.targets
            print(f"Train set shape: {InD_train_set.shape}")

            train_set = np.concatenate((InD_train_set, OoD_train_set), axis=0)
            train_targets = np.concatenate((InD_train_targets, OoD_labels.numpy()), axis=0)
            semi_targets = np.concatenate((np.ones(len(InD_train_targets)), -np.ones(len(OoD_labels))), axis=0)
            print(f"Train set size: {len(train_set)}")
            print(f"Train targets size: {len(train_targets)}")
            print(f"Train semi-targets size: {len(semi_targets)}")
            print(Counter(train_targets))

            # Train set creation
            self.train_set = MyBTNDataset(root=root, transform=transform, target_transform=target_transform, c=c)
            self.train_set.data = train_set
            self.train_set.labels = train_targets
            self.train_set.semi_targets = semi_targets

        # Get test set
        self.test_set = MyBTNDataset(root=root, transform=transform, target_transform=target_transform, c=c)
        self.test_set.data = np.concatenate((ind_test_set, ood_test_set), axis=0)
        self.test_set.labels = np.concatenate((np.zeros(len(ind_test_set)), np.ones(len(ood_test_set))), axis=0)
        print(f"Test set size: {len(self.test_set)}")
        print(f"Test targets: {Counter(self.test_set.labels)}")

class MyBTNDataset(VisionDataset):
    """Torchvision dataset class with patch of __getitem__ method to also return the index of a data sample."""

    def __init__(self,  c=3, *args, **kwargs):
        super(MyBTNDataset, self).__init__(*args, **kwargs)
        self.c = c
        self.semi_targets = None

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        """Override the original method of the CIFAR10 class.
        Args:
            index (int): Index
        Returns:
            triple: (image, target, index) where target is index of the target class.
        """
        # img, target = self.data[index], self.labels[index]
        if self.semi_targets is not None:
            img, target, semi_target = self.data[index], int(self.labels[index]), int(self.semi_targets[index])
        else:
            img, target, semi_target = self.data[index], int(self.labels[index]), 0

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if self.c != 1:
            img = Image.fromarray(img)
        else:
            if type(img) == np.ndarray:
                img = Image.fromarray(img, mode='L')
            else:
                img = Image.fromarray(img.numpy(), mode='L')
        
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, semi_target, index  # only line changed
