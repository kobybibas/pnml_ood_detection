import logging
import sys
import warnings

import torch
# from pytorchcv.model_provider import get_model as ptcv_get_model
from torch import nn
from torch.serialization import SourceChangeWarning
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

from dataset_utils import FeaturesDataset
from model_arch_utils.densenet import DenseNet3
from model_arch_utils.densenet_gram import DenseNet3Gram
from model_arch_utils.resnet import ResNet34
from model_arch_utils.resnet_gram import ResNet34Gram
from save_product_utils import save_products

sys.path.append('./model_arch_utils')

warnings.filterwarnings("ignore", category=SourceChangeWarning)
logger = logging.getLogger(__name__)


def get_model(model_name: str, trainset_name: str) -> torch.nn.Module:
    if model_name == 'densenet':
        if trainset_name in ['cifar10', 'svhn']:
            model = DenseNet3(100, 10)
        elif trainset_name == 'cifar100':
            model = DenseNet3(100, 100)
        else:
            raise ValueError(f'trainset_name={trainset_name} is not supported')
    elif model_name == 'resnet':
        if trainset_name in ['cifar10', 'svhn']:
            model = ResNet34(num_c=10)
        elif trainset_name == 'cifar100':
            model = ResNet34(num_c=100)
        else:
            raise ValueError(f'trainset_name={trainset_name} is not supported')
    else:
        raise ValueError(f'model_name={model_name} is not supported')
    model.load(f'../models/{model_name}_{trainset_name}.pth')
    return model


def get_gram_model(model_name: str, trainset_name: str) -> torch.nn.Module:
    if model_name == 'densenet':
        if trainset_name in ['cifar10', 'svhn']:
            model = DenseNet3Gram(100, 10)
        elif trainset_name == 'cifar100':
            model = DenseNet3Gram(100, 100)
        else:
            raise ValueError(f'trainset_name={trainset_name} is not supported')
    elif model_name == 'resnet':
        if trainset_name in ['cifar10', 'svhn']:
            model = ResNet34Gram(num_c=10)
        elif trainset_name == 'cifar100':
            model = ResNet34Gram(num_c=100)
        else:
            raise ValueError(f'trainset_name={trainset_name} is not supported')
    else:
        raise ValueError(f'model_name={model_name} is not supported')
    model.load(f'../models/{model_name}_{trainset_name}.pth')
    return model


def test_pretrained_model(model: nn.Module, trainloader: data.DataLoader, testloader: data.DataLoader,
                          is_dev_run: bool = False):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss(reduction='sum')
    model.eval()

    with torch.no_grad():
        for data_type, dataloader in zip(['trainset', 'testset'], [trainloader, testloader]):
            loss = 0
            correct = 0
            for images, labels in tqdm(dataloader):
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                loss += loss.item()  # loss sum for all the batch
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

                if is_dev_run is True:
                    break

            acc = correct / len(dataloader.dataset)
            loss /= len(dataloader.dataset)
            logger.info('Pretrained model: {} [Acc Error Loss]=[{:.2f}% {:.2f}% {:.3f}]'.format(
                data_type, 100 * acc, 100 - 100 * acc, loss))


def extract_features_from_loader(model: torch.nn.Module,
                                 dataloader: data.DataLoader, is_dev_run: bool = False) -> FeaturesDataset:
    features_list, labels_list, outputs_list, prob_list = [], [], [], []
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch_num, (images, labels) in enumerate(tqdm(dataloader)):
            # Forward pass
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=-1)

            # Get Features
            features_list.append(model.get_features().cpu().detach())
            labels_list.append(labels.cpu())
            outputs_list.append(outputs.cpu().detach())
            prob_list.append(probs.cpu().detach())

            if is_dev_run is True and batch_num >= 1:
                break
    features_dataset = FeaturesDataset(features_list, labels_list, outputs_list, prob_list,
                                       transform=transforms.Compose([transforms.Lambda(lambda x: x)]))
    torch.cuda.empty_cache()

    return features_dataset


def extract_baseline_features(model, loaders_dict: dict, out_dir: str, is_dev_run: bool = False):
    # Extract features for all dataset: trainset, ind_testset and ood_testsets
    for data_name, loader in loaders_dict.items():
        logger.info('Feature extraction for {}'.format(data_name))
        features_dataset = extract_features_from_loader(model, loader, is_dev_run=is_dev_run)
        save_products(features_dataset, out_dir, data_name)
        logger.info('')
