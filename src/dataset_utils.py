import logging
from glob import glob
from os import path as osp
from os.path import join as osj

import numpy as np
import torch
from PIL import Image
from torch.utils import data
from torchvision import datasets
from torchvision import transforms
from torchvision.datasets.folder import default_loader

logger = logging.getLogger(__name__)
testsets_names = [
    'iSUN',  # iSUN
    'LSUN_resize',  # LSUN (resize)
    'LSUN',  # LSUN (crop)
    'Imagenet_resize',  # Tiny-ImageNet (resize)
    'Imagenet',  # Tiny - ImageNet(crop)
    'Uniform',  # Uniform noise
    'Gaussian'  # Gaussian noise
]


class ImageFolderOOD(datasets.VisionDataset):
    def __init__(self, root, *args, **kwargs):
        super().__init__(root, *args, **kwargs)
        img_path_list = glob(osp.join(root, '*', '*.jpeg')) + \
                        glob(osp.join(root, '*', '*.png')) + \
                        glob(osp.join(root, '*', '*.jpg'))
        if len(img_path_list) == 0:
            logger.error('Dataset was not downloaded {}'.format(root))

        self.data_paths = img_path_list
        self.targets = [-1] * len(img_path_list)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img_path, target = self.data_paths[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = np.array(default_loader(img_path))
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data_paths)


class UniformNoiseDataset(datasets.VisionDataset):
    """
    Create dataset with random noise images in the same structure of CIFAR10
    """

    def __init__(self, root, *args, **kwargs):
        super().__init__(root, *args, **kwargs)
        # Create random data and labels

        self.data = np.random.randint(0, 255, (10000, 32, 32, 3)).astype('uint8')
        self.targets = [-1] * 10000

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class GaussianNoiseDataset(datasets.VisionDataset):
    """
    Create dataset with random noise images in the same structure of CIFAR10
    """

    def __init__(self, root, *args, **kwargs):
        super().__init__(root, *args, **kwargs)
        # Create random data and labels
        self.targets = [-1] * 10000

        self.data = 255 * np.random.randn(10000, 32, 32, 3) + 255 / 2
        self.data = np.clip(self.data, 0, 255).astype('uint8')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class FeaturesDataset(datasets.VisionDataset):

    def __init__(self, features_list: list, labels_list: list, outputs_list: list, prob_list: list, *args, **kwargs):
        super().__init__('', *args, **kwargs)

        self.data = torch.cat(features_list).cpu().numpy()
        self.outputs = torch.cat(outputs_list).cpu().numpy()
        self.targets = torch.cat(labels_list).cpu().numpy()
        self.probs = torch.cat(prob_list).cpu().numpy()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


def get_dataloaders(model_name: str, trainset_name: str, root: str,
                    batch_size: int = 128, n_workers: int = 4, is_training: bool = False,
                    dev_run: bool = False) -> dict:
    assert trainset_name in ['cifar10', 'cifar100', 'svhn']
    if model_name == 'densenet':
        norm_transform = transforms.Normalize((125.3 / 255, 123.0 / 255, 113.9 / 255),
                                              (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0))
    elif model_name == 'resnet':
        norm_transform = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                              (0.2023, 0.1994, 0.2010))
    elif model_name == 'wrn':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        norm_transform = transforms.Normalize(mean, std)
    else:
        raise ValueError(f'{model_name} is not supported')

    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          norm_transform])
    data_transform = transforms.Compose([transforms.CenterCrop(size=(32, 32)),
                                         transforms.ToTensor(),
                                         norm_transform])

    trainloader_cifar10, testloader_cifar10 = get_cifar10_loaders(data_transform, train_transform,
                                                                  root, batch_size, n_workers,
                                                                  is_training)
    trainloader_cifar100, testloader_cifar100 = get_cifar100_loaders(data_transform, train_transform,
                                                                     root, batch_size, n_workers,
                                                                     is_training)
    trainloader_svhn, testloader_svhn = get_svhn_loaders(data_transform, train_transform,
                                                         osj(root, 'svhn'), batch_size, n_workers,
                                                         is_training)
    trainloader_dict = {'cifar10': trainloader_cifar10, 'cifar100': trainloader_cifar100, 'svhn': trainloader_svhn}

    # Load out of distribution datasets
    loaders_dict = {}
    for name in testsets_names:
        if dev_run is True:
            break
        elif name == 'Uniform':
            loaders_dict['Uniform'] = get_uniform_noise_loader(data_transform, batch_size, n_workers)
        elif name == 'Gaussian':
            loaders_dict['Gaussian'] = get_gaussian_noise_loader(data_transform, batch_size, n_workers)
        else:
            loaders_dict[name] = get_image_folder_loader(data_transform, osj(root, name), batch_size, n_workers)

    loaders_dict['svhn'] = testloader_svhn
    loaders_dict['cifar10'] = testloader_cifar10
    loaders_dict['cifar100'] = testloader_cifar100
    loaders_dict['trainset'] = trainloader_dict[trainset_name]
    return loaders_dict


def get_cifar10_loaders(data_transform, train_transform,
                        data_dir: str = './data', batch_size: int = 128, num_workers: int = 4,
                        is_training: bool = False):
    """
    create train and test pytorch dataloaders for CIFAR10 dataset
    :param data_dir: the folder that will contain the data
    :param batch_size: the size of the batch for test and train loaders
    :param num_workers: number of cpu workers which loads the GPU with the dataset
    :return: train and test loaders along with mapping between labels and class names
    """
    trainset = datasets.CIFAR10(root=data_dir,
                                train=True,
                                download=False,
                                transform=data_transform if is_training is False else train_transform)
    trainloader = data.DataLoader(trainset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  pin_memory=True
                                  )

    testset = datasets.CIFAR10(root=data_dir,
                               train=False,
                               download=False,
                               transform=data_transform)
    testloader = data.DataLoader(testset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 pin_memory=True
                                 )
    return trainloader, testloader


def get_cifar100_loaders(data_transform, train_transform,
                         data_dir: str = './data', batch_size: int = 128, num_workers: int = 4,
                         is_training: bool = False):
    """
    create train and test pytorch dataloaders for CIFAR100 dataset
    :param data_dir: the folder that will contain the data
    :param batch_size: the size of the batch for test and train loaders
    :param num_workers: number of cpu workers which loads the GPU with the dataset
    :return: train and test loaders along with mapping between labels and class names
    """
    trainset = datasets.CIFAR100(root=data_dir,
                                 train=True,
                                 download=False,
                                 transform=data_transform if is_training is False else train_transform)
    trainloader = data.DataLoader(trainset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  pin_memory=True
                                  )

    testset = datasets.CIFAR100(root=data_dir,
                                train=False,
                                download=False,
                                transform=data_transform)
    testloader = data.DataLoader(testset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 pin_memory=True
                                 )
    return trainloader, testloader


def get_svhn_loaders(data_transform, train_transform,
                     data_dir: str = './data', batch_size: int = 128, num_workers: int = 4,
                     is_training: bool = False):
    """
    create train and test pytorch dataloaders for CIFAR100 dataset
    :param data_dir: the folder that will contain the data
    :param batch_size: the size of the batch for test and train loaders
    :param num_workers: number of cpu workers which loads the GPU with the dataset
    :return: train and test loaders along with mapping between labels and class names
    """
    trainset = datasets.SVHN(root=data_dir,
                             split='train',
                             download=False,
                             transform=data_transform if is_training is False else train_transform)
    trainloader = data.DataLoader(trainset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  pin_memory=True
                                  )

    testset = datasets.SVHN(root=data_dir,
                            split='test',
                            download=False,
                            transform=data_transform)
    testloader = data.DataLoader(testset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 pin_memory=True
                                 )
    return trainloader, testloader


def get_image_folder_loader(data_transform, path: str, batch_size: int = 128, num_workers: int = 4):
    testset = ImageFolderOOD(root=path, transform=data_transform)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=num_workers,
                                             pin_memory=True
                                             )
    return testloader


def get_uniform_noise_loader(data_transform, batch_size: int = 128, num_workers: int = 4):
    """
    create trainloader for CIFAR10 dataset and testloader with noise images
    :param batch_size: the size of the batch for test and train loaders
    :param num_workers: number of cpu workers which loads the GPU with the dataset
    :return: train and test loaders along with mapping between labels and class names
    """
    testset = UniformNoiseDataset(root='', transform=data_transform)
    testloader = data.DataLoader(testset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 pin_memory=True
                                 )
    return testloader


def get_gaussian_noise_loader(data_transform, batch_size: int = 128, num_workers: int = 4):
    """
    create trainloader for CIFAR10 dataset and testloader with noise images
    :param batch_size: the size of the batch for test and train loaders
    :param num_workers: number of cpu workers which loads the GPU with the dataset
    :return: train and test loaders along with mapping between labels and class names
    """
    testset = GaussianNoiseDataset(root='', transform=data_transform)
    testloader = data.DataLoader(testset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 pin_memory=True
                                 )
    return testloader
