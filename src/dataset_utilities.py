import copy
import os
import os.path
from glob import glob
from time import time

import numpy as np
import torch
from PIL import Image
from loguru import logger
from torch.utils import data
from torchvision import transforms, datasets
from torchvision.datasets.folder import default_loader
from tqdm import tqdm

from adversarial_utilities import create_adversarial_sign_dataset
from dataset_classes import NoiseDataset

# Normalization for CIFAR10 dataset
# mean_cifar10 = [0.485, 0.456, 0.406]
# std_cifar10 = [0.229, 0.224, 0.225]
# normalize = transforms.Normalize(mean=mean_cifar10, std=std_cifar10)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((125.3 / 255, 123.0 / 255, 113.9 / 255), (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0)),
])


def insert_sample_to_dataset(trainloader_org,
                             trainset_empty,
                             data_single, label_single,
                             batches_num: int = -1):
    """
    Inserting test sample into the trainset
    :param trainloader: contains the trainset
    :param data_single: the data which we want to insert
    :param label_single: the data label which we want to insert
    :return: dataloader which contains the trainset with the additional sample
    """
    time_start = time()
    dataset_train_org = trainloader_org.dataset
    dataset_train = copy.deepcopy(trainset_empty)
    dataset_single_sample = copy.deepcopy(trainset_empty)

    if isinstance(data_single, np.ndarray):
        dataset_single_sample.data = np.expand_dims(data_single, 0)
    elif isinstance(data_single, torch.Tensor):
        dataset_single_sample.data = data_single.unsqueeze(0)
    else:
        ValueError('Unexpected type {}'.format(type(dataset_single_sample.data)))

    dataset_single_sample.targets = [label_single]

    if batches_num <= 1:
        dataset_train.data = dataset_train_org.data[-trainloader_org.batch_size * batches_num + 1:]
        dataset_train.targets = dataset_train_org.targets[-trainloader_org.batch_size * batches_num + 1:]
    else:
        dataset_train.data = dataset_train_org.data
        dataset_train.targets = dataset_train_org.targets

    trainloader_with_sample = data.DataLoader(data.ConcatDataset([dataset_train, dataset_single_sample]),
                                              batch_size=trainloader_org.batch_size,
                                              shuffle=True,
                                              num_workers=trainloader_org.num_workers)
    # logger.debug(
    #     'Added test sample to train. Trainset size {} in {:.2f} sec'.format(len(trainloader_with_sample.dataset),
    #                                                                         time() - time_start))

    return trainloader_with_sample


def create_svhn_dataloaders(data_dir: str = './data', batch_size: int = 128, num_workers: int = 4):
    """
    create train and test pytorch dataloaders for SVHN dataset
    :param data_dir: the folder that will contain the data
    :param batch_size: the size of the batch for test and train loaders
    :param num_workers: number of cpu workers which loads the GPU with the dataset
    :return: train and test loaders along with mapping between labels and class names
    """

    trainset = datasets.CIFAR10(root=data_dir,
                                train=True,
                                download=True,
                                transform=transform)
    trainloader = data.DataLoader(trainset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers)

    data_dir = os.path.join(data_dir, 'svhn')
    testset = datasets.SVHN(root=data_dir,
                            split='test',
                            download=True,
                            transform=transform)

    # Align as CIFAR10 dataset
    testset.test_data = testset.data
    testset.test_labels = testset.labels

    testloader = data.DataLoader(testset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers)

    # Classes name
    classes_cifar10 = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    classes_svhn = ('1', '2', '3', '4', '5', '6', '7', '8', '9', '0')

    return trainloader, testloader, classes_svhn, classes_cifar10


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
                                transform=transform)
    trainloader = data.DataLoader(trainset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers)

    testset = datasets.CIFAR10(root=data_dir,
                               train=False,
                               download=True,
                               transform=transform)
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
                                 transform=transform)
    trainloader = data.DataLoader(trainset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers)

    testset = datasets.CIFAR100(root=data_dir,
                                train=False,
                                download=True,
                                transform=transform)
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


class ImageFolderOOD(datasets.VisionDataset):
    def __init__(self, root, *args, **kwargs):
        super().__init__(root, *args, **kwargs)
        img_path_list = glob(os.path.join(root, '*', '*.jpeg')) + \
                        glob(os.path.join(root, '*', '*.png')) + \
                        glob(os.path.join(root, '*', '*.jpg'))
        if len(img_path_list) == 0:
            logger.error('Dataset was not downloaded.')
            ValueError('Failed on ImageFolderOOD')

        img_list = []
        for img_path in img_path_list:
            img = default_loader(img_path)
            img_list.append(np.array(img))

        self.data = np.asarray(img_list)
        self.targets = [-1] * len(img_path_list)

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


def create_image_folder_trainloader(path: str, batch_size: int = 128, num_workers: int = 4):
    testset = ImageFolderOOD(root=path,
                             transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)
    return testloader


def generate_noise_sample():
    random_sample_data = np.random.randint(256, size=(32, 32, 3), dtype='uint8')
    random_sample_label = -1
    return random_sample_data, random_sample_label


class CIFAR10RandomLabels(datasets.CIFAR10):
    """CIFAR10 dataset, with support for randomly corrupt labels.

    Params
    ------
    corrupt_prob: float
        Default 0.0. The probability of a label being replaced with
        random label.
    num_classes: int
        Default 10. The number of classes in the dataset.
    """

    def __init__(self, corrupt_prob=0.0, num_classes=10, **kwargs):
        super(CIFAR10RandomLabels, self).__init__(**kwargs)
        self.n_classes = num_classes
        if corrupt_prob > 0:
            self.corrupt_labels(corrupt_prob)

    def corrupt_labels(self, corrupt_prob):
        labels = np.array(self.train_labels if self.train else self.test_labels)
        np.random.seed(12345)
        mask = np.random.rand(len(labels)) <= corrupt_prob
        rnd_labels = np.random.choice(self.n_classes, mask.sum())
        labels[mask] = rnd_labels
        # we need to explicitly cast the labels from npy.int64 to
        # builtin int type, otherwise pytorch will fail...
        labels = [int(x) for x in labels]

        if self.train:
            self.train_labels = labels
        else:
            self.test_labels = labels


def create_cifar10_random_label_dataloaders(data_dir: str = './data', batch_size: int = 128, num_workers: int = 4,
                                            label_corrupt_prob=1.0):
    """
    create train and test pytorch dataloaders for CIFAR10 dataset.
    Train set can be with random labels, the probability to be random depends on label_corrupt_prob.
    :param data_dir: the folder that will contain the data.
    :param batch_size: the size of the batch for test and train loaders.
    :param label_corrupt_prob: the probability to be random of label of train sample.
    :param num_workers: number of cpu workers which loads the GPU with the dataset.
    :return: train and test loaders along with mapping between labels and class names.
    """

    # Trainset with random labels
    trainset = CIFAR10RandomLabels(root=data_dir,
                                   train=True,
                                   download=True,
                                   transform=transform,
                                   corrupt_prob=label_corrupt_prob)
    trainloader = data.DataLoader(trainset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers)

    # Testset with real labels
    testset = datasets.CIFAR10(root=data_dir,
                               train=False,
                               download=True,
                               transform=transform)
    testloader = data.DataLoader(testset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes


def dataloaders_noise(data_dir: str = './data', batch_size: int = 128, num_workers: int = 4):
    """
    create trainloader for CIFAR10 dataset and testloader with noise images
    :param data_dir: the folder that will contain the data
    :param batch_size: the size of the batch for test and train loaders
    :param num_workers: number of cpu workers which loads the GPU with the dataset
    :return: train and test loaders along with mapping between labels and class names
    """
    testset = NoiseDataset(root=data_dir,
                           transform=transform)
    testloader = data.DataLoader(testset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers)
    return testloader


class CIFAR10Adversarial(datasets.CIFAR10):
    """
    Implementing adversarial attack to CIFAR10 testset.
    """

    def __init__(self, epsilon=0.005, adversarial_sign_dataset_path='./data/adversarial_sign', **kwargs):
        """

        :param epsilon: the strength of the attack. Fast gradient sign attack.
        :param adversarial_sign_dataset_path: path in which the gradients sign from the back propagation is saved into.
        :param kwargs: initial init arguments.
        """
        super(CIFAR10Adversarial, self).__init__(**kwargs)
        self.adversarial_sign_dataset_path = adversarial_sign_dataset_path
        self.epsilon = epsilon
        for index in range(self.test_data.shape[0]):
            sign = np.load(os.path.join(self.adversarial_sign_dataset_path, str(index) + '.npy'))
            sign = np.transpose(sign, (1, 2, 0))
            self.test_data[index] = np.clip(self.test_data[index] + (epsilon * 255) * sign, 0, 255)


def create_adversarial_cifar10_dataloaders(data_dir: str = './data',
                                           adversarial_dir: str = os.path.join('data',
                                                                               'adversarial_sign'),
                                           epsilon: float = 0.05,
                                           batch_size: int = 128,
                                           num_workers: int = 4):
    """
    create train and test pytorch dataloaders for CIFAR10 dataset
    :param data_dir: the folder that will contain the data
    :param adversarial_dir: the output dir to which the gradient adversarial sign will be saved.
    :param epsilon: the additive gradient strength to be added to the image.
    :param batch_size: the size of the batch for test and train loaders
    :param num_workers: number of cpu workers which loads the GPU with the dataset
    :return: train and test loaders along with mapping between labels and class names
    """
    trainset = datasets.CIFAR10(root=data_dir,
                                train=True,
                                download=True,
                                transform=transform)
    trainloader = data.DataLoader(trainset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers)

    adversarial_sign_dataset_path = create_adversarial_sign_dataset(data_dir, output_folder=adversarial_dir)
    testset = CIFAR10Adversarial(root=data_dir,
                                 train=False,
                                 download=True,
                                 transform=transform,
                                 adversarial_sign_dataset_path=adversarial_sign_dataset_path,
                                 epsilon=epsilon)
    testloader = data.DataLoader(testset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes


class FeaturesDataset(datasets.VisionDataset):

    def __init__(self, features_list: list, labels_list: list, *args, **kwargs):
        super().__init__('', *args, **kwargs)

        self.data = torch.cat(features_list).cpu()
        self.targets = torch.cat(labels_list).cpu().numpy().tolist()

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


def extract_features(model, dataloaders: dict, is_preprocess: bool = False,
                     temper: float = 1000.0,
                     noise_magnitude: float = 0.0014):
    # Extract features from train and test

    model.convert_to_feature_extractor()
    model = model.cuda() if torch.cuda.is_available() else model
    features_datasets_dict = {}
    for set_type in ['train', 'test']:
        if set_type == 'test' and is_preprocess is True:
            continue
        time_start = time()

        features_list = []
        labels_list = []

        for images, labels in tqdm(dataloaders[set_type]):
            images = images.cuda() if torch.cuda.is_available() else images
            labels = labels.cuda() if torch.cuda.is_available() else labels

            features = model(images)

            features_list.append(features.cpu())
            labels_list.append(labels.cpu())

        features_datasets_dict[set_type] = FeaturesDataset(features_list,
                                                           labels_list,
                                                           transform=transforms.Compose([
                                                               transforms.Lambda(lambda x: x)]))
        torch.cuda.empty_cache()
        logger.debug('feature extraction {} in {:.2f}'.format(set_type, time() - time_start))

    if is_preprocess is True:
        norm = [63.0 / 255.0, 62.1 / 255.0, 66.7 / 255.0]
        features_list, labels_list = [], []
        criterion = torch.nn.CrossEntropyLoss()
        time_start = time()
        for inputs, labels_org in tqdm(dataloaders['test']):
            inputs = inputs.cuda() if torch.cuda.is_available() else inputs
            inputs = inputs.requires_grad_()
            outputs = model.forward_super(inputs)

            # Using temperature scaling
            outputs = outputs / temper

            # Calculating the perturbation we need to add, that is,
            # the sign of gradient of cross entropy loss w.r.t. input
            labels = torch.argmax(outputs, axis=1)
            loss = criterion(outputs, labels)
            loss.backward()

            # Normalizing the gradient to binary in {0, 1}
            gradient = torch.ge(inputs.grad.data, 0)
            gradient = (gradient.float() - 0.5) * 2

            # Normalizing the gradient to the same space of image
            gradient[:, 0] = (gradient[:, 0]) / (norm[0])
            gradient[:, 1] = (gradient[:, 1]) / (norm[1])
            gradient[:, 2] = (gradient[:, 2]) / (norm[2])
            # Adding small perturbations to images

            inputs_temp = torch.add(inputs.data, -noise_magnitude, gradient)
            features = model(inputs_temp)

            features_list.append(features.cpu())
            labels_list.append(labels_org.cpu())

        features_datasets_dict['test'] = FeaturesDataset(features_list,
                                                         labels_list,
                                                         transform=transforms.Compose([
                                                             transforms.Lambda(lambda x: x)]))

        logger.debug('feature extraction with is_preprocess. {:.2f} sec'.format(time() - time_start))

    torch.cuda.empty_cache()
    model.convert_to_classifier()

    return features_datasets_dict['train'], features_datasets_dict['test']
