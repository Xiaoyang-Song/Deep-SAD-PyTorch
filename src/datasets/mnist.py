from torch.utils.data import Subset
from PIL import Image
from torchvision.datasets import MNIST
from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import create_semisupervised_setting, create_semisupervised_setting_number, get_target_label_idx
from collections import Counter
import torch
import torchvision.transforms as transforms
import random
import numpy as np
import os


class MNIST_Dataset_Customized(TorchvisionDataset):

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
        train_set = MyMNIST(root=self.root, train=True, transform=transform, target_transform=target_transform, download=True)
        print(f"Original training data size: ", len(train_set), train_set.data.shape)

        if n_known_outlier_classes == 0:
            self.known_outlier_classes = ()
        elif n_known_outlier_classes == 1:
            self.known_outlier_classes = tuple([known_outlier_class])
        else:
            # self.known_outlier_classes = tuple(random.sample(self.outlier_classes, n_known_outlier_classes))
            self.known_outlier_classes = self.outlier_classes


        # Create semi-supervised setting
        if sampler == "original":
            print("Using original sampler")
            assert ratio_known_normal is not None and ratio_known_outlier is not None \
                and ratio_pollution is not None and n_known_outlier_classes is not None, \
                "If sampler is 'original', ratio_known_normal, ratio_known_outlier, and ratio_pollution must be provided."
            
            idx, _, semi_targets = create_semisupervised_setting(train_set.targets.cpu().data.numpy(), self.normal_classes,
                                                                self.outlier_classes, self.known_outlier_classes,
                                                                ratio_known_normal, ratio_known_outlier, ratio_pollution)
            print(len(idx), len(semi_targets))
            train_set.semi_targets[idx] = torch.tensor(semi_targets)  # set respective semi-supervised labels
            self.train_set = Subset(train_set, idx)

        elif sampler == "number-random":
            print("Using number-random sampler")
            assert n_known_normal is not None and n_known_outlier is not None and n_pollution is not None, \
                "If sampler is 'number-random', n_known_normal, n_known_outlier, and n_pollution must be provided."
            n_known_normal = int(len(train_set))
            n_pollution = 0 # assume no pollution
            idx, _, semi_targets = create_semisupervised_setting_number(train_set.targets.cpu().data.numpy(), self.normal_classes,
                                                                self.outlier_classes, self.known_outlier_classes,
                                                                n_known_normal, n_known_outlier, n_pollution)
            print(len(idx), len(semi_targets))
            train_set.semi_targets[idx] = torch.tensor(semi_targets)  # set respective semi-supervised labels
            self.train_set = Subset(train_set, idx)

        elif sampler == "number-pre-sampled":
            assert regime is not None and n_known_outlier is not None, \
                "If sampler is 'number-pre-sampled', regime and n_outlier must be provided."
            print("Using number-pre-sampled sampler")
            train_idx_normal = get_target_label_idx(train_set.targets, self.normal_classes)
            # InD
            InD_train_set = train_set.data[train_idx_normal]
            InD_train_targets = train_set.targets[train_idx_normal]
            print(f"Train set shape: {InD_train_set.shape}")

            # OOD from pre-sampled data
            OoD_path = os.path.join("..", "..", "Out-of-Distribution-GANs", "checkpoint", "OOD-Sample", "FashionMNIST", f"OOD-{regime}-{n_known_outlier}.pt")
            OoD_data, OoD_labels = torch.load(OoD_path)
            OoD_train_set = np.array(OoD_data.squeeze())

            train_set = torch.tensor(np.concatenate((InD_train_set, OoD_train_set), axis=0))
            train_targets = np.concatenate((InD_train_targets, OoD_labels.numpy()), axis=0)
            semi_targets = np.concatenate((np.ones(len(InD_train_targets)), -np.ones(len(OoD_labels))), axis=0)
            print(f"Train set size: {len(train_set)}")
            print(f"Train targets size: {len(train_targets)}")
            print(f"Train semi-targets size: {len(semi_targets)}")
            print(Counter(train_targets))

            self.train_set = MyMNIST(root=self.root, train=True, transform=transform, target_transform=target_transform, download=True)
            self.train_set.data = train_set
            self.train_set.targets = train_targets
            self.train_set.semi_targets = semi_targets

        # Get test set
        self.test_set = MyMNIST(root=self.root, train=False, transform=transform, target_transform=target_transform, download=True)



class MNIST_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class: int = 0, known_outlier_class: int = 1, n_known_outlier_classes: int = 0,
                 ratio_known_normal: float = 0.0, ratio_known_outlier: float = 0.0, ratio_pollution: float = 0.0):
        super().__init__(root)

        # Define normal and outlier classes
        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 10))
        # self.outlier_classes.remove(normal_class)
        # self.outlier_classes = tuple(self.outlier_classes)
        if type(normal_class) == int:
            self.outlier_classes.remove(normal_class)
        else:
            self.outlier_classes = list(set(self.outlier_classes) - set(normal_class))
        print(f"Outlier classes: {self.outlier_classes}")

        if n_known_outlier_classes == 0:
            self.known_outlier_classes = ()
        elif n_known_outlier_classes == 1:
            self.known_outlier_classes = tuple([known_outlier_class])
        else:
            # self.known_outlier_classes = tuple(random.sample(self.outlier_classes, n_known_outlier_classes))
            self.known_outlier_classes = self.outlier_classes
            

        # MNIST preprocessing: feature scaling to [0, 1]
        transform = transforms.ToTensor()
        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        # Get train set
        train_set = MyMNIST(root=self.root, train=True, transform=transform, target_transform=target_transform, download=True)
        print(len(train_set))

        # Create semi-supervised setting
        # idx, _, semi_targets = create_semisupervised_setting(train_set.targets.cpu().data.numpy(), self.normal_classes,
        #                                                      self.outlier_classes, self.known_outlier_classes,
        #                                                      ratio_known_normal, ratio_known_outlier, ratio_pollution)
        n_known_normal = int(len(train_set))
        n_known_outlier = 1000
        n_pollution = 0 # assume no pollution
        idx, _, semi_targets = create_semisupervised_setting_number(train_set.targets.cpu().data.numpy(), self.normal_classes,
                                                             self.outlier_classes, self.known_outlier_classes,
                                                             n_known_normal, n_known_outlier, n_pollution)
        print(len(idx), len(semi_targets))
        train_set.semi_targets[idx] = torch.tensor(semi_targets)  # set respective semi-supervised labels


        print(Counter(np.array(train_set.targets)[idx]))
        # Subset train_set to semi-supervised setup
        self.train_set = Subset(train_set, idx)
        print(len(self.train_set))

        # Get test set
        self.test_set = MyMNIST(root=self.root, train=False, transform=transform, target_transform=target_transform, download=True)


class MyMNIST(MNIST):
    """
    Torchvision MNIST class with additional targets for the semi-supervised setting and patch of __getitem__ method
    to also return the semi-supervised target as well as the index of a data sample.
    """

    def __init__(self, *args, **kwargs):
        super(MyMNIST, self).__init__(*args, **kwargs)

        self.semi_targets = torch.zeros_like(self.targets)

    def __getitem__(self, index):
        """Override the original method of the MNIST class.
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, semi_target, index)
        """
        img, target, semi_target = self.data[index], int(self.targets[index]), int(self.semi_targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, semi_target, index
