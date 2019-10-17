import os.path as osp

import torch
from torch.utils import data
from torchvision import transforms, datasets

from dataset_classes import UniformNoiseDataset, GaussianNoiseDataset, ImageFolderOOD

testsets_name = [
    # Tiny - ImageNet(crop)
    'Imagenet',
    # Tiny-ImageNet (resize)
    'Imagenet_resize',
    # LSUN (crop)
    'LSUN',
    # LSUN (resize)
    'LSUN_resize',
    # iSUN
    'iSUN',
    # Gaussian noise
    'Gaussian',
    # Uniform noise
    'Uniform'
]

# Cifar10
mean_rgb_cifar = (0.4914, 0.4822, 0.4465)
std_rgb_cifar = (0.2023, 0.1994, 0.2010)
transform_cifar_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean_rgb_cifar, std=std_rgb_cifar),
])

transform_cifar_train = transforms.Compose([
    transforms.RandomCrop(size=32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean_rgb_cifar, std=std_rgb_cifar),
])


def get_dataloaders(trainset_name: str, data_dir: str, batch_size: int = 128, num_workers: int = 4) -> dict:
    assert trainset_name in ['cifar10', 'cifar100']

    trainloader_cifar10, testloader_cifar10, _ = create_cifar10_dataloaders(data_dir, batch_size, num_workers)
    trainloader_cifar100, testloader_cifar100, _ = create_cifar100_dataloaders(data_dir, batch_size, num_workers)

    # Load out of distribution datasets
    dataloaders_dict = {'ood': {}}
    for name in testsets_name:
        if name == 'Uniform':
            dataloaders_dict['ood']['Uniform'] = create_uniform_noise_dataloaders(batch_size, num_workers)
        elif name == 'Gaussian':
            dataloaders_dict['ood']['Gaussian'] = create_gaussian_noise_dataloaders(batch_size, num_workers)
        else:
            dataloaders_dict['ood'][name] = create_image_folder_trainloader(osp.join(data_dir, name), batch_size,
                                                                            num_workers)
        dataloaders_dict['ood']['cifar100'] = testloader_cifar100
        dataloaders_dict['ood']['cifar10'] = testloader_cifar10

    dataloaders_dict['trainset'] = trainloader_cifar10 if trainset_name == 'cifar10' else trainloader_cifar100
    dataloaders_dict['testset'] = testloader_cifar10 if trainset_name == 'cifar10' else testloader_cifar100
    return dataloaders_dict


def create_cifar10_dataloaders(data_dir: str = './data', batch_size: int = 128, num_workers: int = 4):
    """
    create train and test pytorch dataloaders for CIFAR10 dataset
    :param data_dir: the folder that will contain the data
    :param batch_size: the size of the batch for test and train loaders
    :param num_workers: number of cpu workers which loads the GPU with the dataset
    :return: train and test loaders along with mapping between labels and class names
    """
    trainset = datasets.CIFAR10(root=data_dir,
                                train=True,
                                download=True,
                                transform=transform_cifar_test)
    trainloader = data.DataLoader(trainset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers)

    testset = datasets.CIFAR10(root=data_dir,
                               train=False,
                               download=True,
                               transform=transform_cifar_test)
    testloader = data.DataLoader(testset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes


def create_cifar100_dataloaders(data_dir: str = './data', batch_size: int = 128, num_workers: int = 4):
    """
    create train and test pytorch dataloaders for CIFAR100 dataset
    :param data_dir: the folder that will contain the data
    :param batch_size: the size of the batch for test and train loaders
    :param num_workers: number of cpu workers which loads the GPU with the dataset
    :return: train and test loaders along with mapping between labels and class names
    """
    trainset = datasets.CIFAR100(root=data_dir,
                                 train=True,
                                 download=True,
                                 transform=transform_cifar_test)
    trainloader = data.DataLoader(trainset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers)

    testset = datasets.CIFAR100(root=data_dir,
                                train=False,
                                download=True,
                                transform=transform_cifar_test)
    testloader = data.DataLoader(testset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers)
    classes = ('apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
               'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
               'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
               'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
               'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
               'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
               'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
               'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
               'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
               'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
               'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
               'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
               'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
               'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
               'worm')

    return trainloader, testloader, classes


def create_image_folder_trainloader(path: str, batch_size: int = 128, num_workers: int = 4):
    testset = ImageFolderOOD(root=path, transform=transform_cifar_test)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=num_workers)
    return testloader


def create_uniform_noise_dataloaders(batch_size: int = 128, num_workers: int = 4):
    """
    create trainloader for CIFAR10 dataset and testloader with noise images
    :param data_dir: the folder that will contain the data
    :param batch_size: the size of the batch for test and train loaders
    :param num_workers: number of cpu workers which loads the GPU with the dataset
    :return: train and test loaders along with mapping between labels and class names
    """
    testset = UniformNoiseDataset(root='', transform=transform_cifar_test)
    testloader = data.DataLoader(testset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers)
    return testloader


def create_gaussian_noise_dataloaders(batch_size: int = 128, num_workers: int = 4):
    """
    create trainloader for CIFAR10 dataset and testloader with noise images
    :param data_dir: the folder that will contain the data
    :param batch_size: the size of the batch for test and train loaders
    :param num_workers: number of cpu workers which loads the GPU with the dataset
    :return: train and test loaders along with mapping between labels and class names
    """
    testset = GaussianNoiseDataset(root='', transform=transform_cifar_test)
    testloader = data.DataLoader(testset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers)
    return testloader
