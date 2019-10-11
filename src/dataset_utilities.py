import torch
from torch.utils import data
from torchvision import transforms, datasets

from dataset_classes import UniformNoiseDataset, GaussianNoiseDataset, ImageFolderOOD

mean_rgb = (0.4914, 0.4822, 0.4465)
std_rgb = (0.2023, 0.1994, 0.2010)
jitter_param = 0.4

transform_cifar_train = transforms.Compose([
    transforms.RandomCrop(size=32, padding=4),
    transforms.RandomHorizontalFlip(),
    # transforms.ColorJitter(
    #     brightness=jitter_param,
    #     contrast=jitter_param,
    #     saturation=jitter_param),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=mean_rgb,
        std=std_rgb),
])

transform_cifar_test = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize(mean_rgb, std_rgb),
])
transform_noise = transforms.Compose([
    transforms.Normalize(mean_rgb, std_rgb),
])


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
    testset = ImageFolderOOD(root=path,
                             transform=transform_cifar_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)
    return testloader


def create_uniform_noise_dataloaders(batch_size: int = 128, num_workers: int = 4):
    """
    create trainloader for CIFAR10 dataset and testloader with noise images
    :param data_dir: the folder that will contain the data
    :param batch_size: the size of the batch for test and train loaders
    :param num_workers: number of cpu workers which loads the GPU with the dataset
    :return: train and test loaders along with mapping between labels and class names
    """
    testset = UniformNoiseDataset(root='',
                                  transform=transform_cifar_test)
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
    testset = GaussianNoiseDataset(root='',
                                   transform=transform_cifar_test)
    testloader = data.DataLoader(testset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers)
    return testloader
