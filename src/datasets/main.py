from .mnist import MNIST_Dataset, MNIST_Dataset_Customized
from .fmnist import FashionMNIST_Dataset, FashionMNIST_Dataset_Customized
from .cifar10 import CIFAR10_Dataset
from .btn_dset import BTN_Dataset
from .odds import ODDSADDataset


def load_dataset(dataset_name, data_path, normal_class, known_outlier_class, n_known_outlier_classes: int = 0,
                 ratio_known_normal: float = 0.0, ratio_known_outlier: float = 0.0, ratio_pollution: float = 0.0,
                 n_known_normal=None, n_known_outlier=None, n_pollution=None, sampler="original",regime=None,
                 random_state=None):
    """Loads the dataset."""

    implemented_datasets = ('mnist', 'fmnist', 'cifar10',
                            'cifar10-svhn', 'mnist-fashionmnist',
                            'arrhythmia', 'cardio', 'satellite', 'satimage-2', 'shuttle', 'thyroid')
    assert dataset_name in implemented_datasets

    dataset = None

    if dataset_name == 'mnist':
        # dataset = MNIST_Dataset(root=data_path,
        #                         normal_class=normal_class,
        #                         known_outlier_class=known_outlier_class,
        #                         n_known_outlier_classes=n_known_outlier_classes,
        #                         ratio_known_normal=ratio_known_normal,
        #                         ratio_known_outlier=ratio_known_outlier,
        #                         ratio_pollution=ratio_pollution)

        dataset = MNIST_Dataset_Customized(root=data_path,
                                           normal_class=normal_class,
                                           known_outlier_class=known_outlier_class,
                                           n_known_outlier_classes=n_known_outlier_classes,
                                           ratio_known_normal=ratio_known_normal,
                                           ratio_known_outlier=ratio_known_outlier,
                                           ratio_pollution=ratio_pollution,
                                           n_known_normal=n_known_normal,
                                           n_known_outlier=n_known_outlier,
                                           n_pollution=n_pollution,
                                           sampler=sampler,
                                           regime=regime)
    if dataset_name == 'fmnist':
        # dataset = FashionMNIST_Dataset(root=data_path,
        #                                normal_class=normal_class,
        #                                known_outlier_class=known_outlier_class,
        #                                n_known_outlier_classes=n_known_outlier_classes,
        #                                ratio_known_normal=ratio_known_normal,
        #                                ratio_known_outlier=ratio_known_outlier,
        #                                ratio_pollution=ratio_pollution)

        dataset = FashionMNIST_Dataset_Customized(root=data_path,
                                           normal_class=normal_class,
                                           known_outlier_class=known_outlier_class,
                                           n_known_outlier_classes=n_known_outlier_classes,
                                           ratio_known_normal=ratio_known_normal,
                                           ratio_known_outlier=ratio_known_outlier,
                                           ratio_pollution=ratio_pollution,
                                           n_known_normal=n_known_normal,
                                           n_known_outlier=n_known_outlier,
                                           n_pollution=n_pollution,
                                           sampler=sampler,
                                           regime=regime)
        
    if dataset_name == 'cifar10-svhn':
        dataset = BTN_Dataset(root=data_path,
                            normal_class=normal_class,
                            known_outlier_class=known_outlier_class,
                            n_known_outlier_classes=n_known_outlier_classes,
                            ratio_known_normal=ratio_known_normal,
                            ratio_known_outlier=ratio_known_outlier,
                            ratio_pollution=ratio_pollution,
                            n_known_normal=n_known_normal,
                            n_known_outlier=n_known_outlier,
                            n_pollution=n_pollution,
                            sampler=sampler,
                            regime=regime,
                            ind = 'cifar10',
                            ood = 'svhn')
        
    if dataset_name == 'cifar10':
        dataset = CIFAR10_Dataset(root=data_path,
                                  normal_class=normal_class,
                                  known_outlier_class=known_outlier_class,
                                  n_known_outlier_classes=n_known_outlier_classes,
                                  ratio_known_normal=ratio_known_normal,
                                  ratio_known_outlier=ratio_known_outlier,
                                  ratio_pollution=ratio_pollution)

    if dataset_name in ('arrhythmia', 'cardio', 'satellite', 'satimage-2', 'shuttle', 'thyroid'):
        dataset = ODDSADDataset(root=data_path,
                                dataset_name=dataset_name,
                                n_known_outlier_classes=n_known_outlier_classes,
                                ratio_known_normal=ratio_known_normal,
                                ratio_known_outlier=ratio_known_outlier,
                                ratio_pollution=ratio_pollution,
                                random_state=random_state)

    return dataset
