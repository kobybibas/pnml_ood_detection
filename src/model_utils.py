import logging
import sys
import types

import torch
# from pytorchcv.model_provider import get_model as ptcv_get_model
from torch import nn
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

from dataset_utils import FeaturesDataset
from model_arch_utils.densenet import DenseNet3
from model_arch_utils.resnet import ResNet34
from model_arch_utils.wideresnet import WideResNet
from odin_utils import odin_extract_features_from_loader

sys.path.append('./model_arch_utils')
logger = logging.getLogger(__name__)


def add_feature_extractor_method(model: torch.nn.Module):
    """
    Add feature extractor method
    :param model: Pytorch model
    :return: In place: model with feature extraction method
    """

    def my_forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        self.features_out = x.clone()
        x = self.output(x)
        return x

    def get_features(self):
        return self.features_out

    model.forward = types.MethodType(my_forward, model)
    model.get_features = types.MethodType(get_features, model)


def get_model(model_name: str, trainset_name: str) -> torch.nn.Module:
    # Get list of pretrained models
    if model_name == 'densenet':
        # model = ptcv_get_model("densenet100_k12_bc_%s" % trainset_name, pretrained=True)
        if trainset_name == 'cifar10':
            model = DenseNet3(100, 10)
            model.load('../models/densenet_cifar10.pth')
        elif trainset_name == 'cifar100':
            model = DenseNet3(100, 100)
            model.load('../models/densenet_cifar100.pth')
        elif trainset_name == 'svhn':
            model = DenseNet3(100, 10)
            model.load('../models/densenet_svhn.pth')
    elif model_name == 'wideresnet':
        if trainset_name == 'cifar10':
            model = WideResNet(100, 10)
            model.load('../models/wideresnet10.pth')
        elif trainset_name == 'cifar100':
            model = WideResNet(100, 100)
            model.load('../models/wideresnet100.pth')
        else:
            raise ValueError(f'{trainset_name} is not supported for {model_name}')
    elif model_name == 'resnet':
        # model = ptcv_get_model("wrn28_10_%s" % trainset_name, pretrained=True)
        if trainset_name == 'cifar10':
            model = ResNet34(num_c=10)
            model.load('../models/resnet_cifar10.pth')
        elif trainset_name == 'cifar100':
            model = ResNet34(num_c=100)
            model.load('../models/resnet_cifar100.pth')
        elif trainset_name == 'svhn':
            model = ResNet34(num_c=10)
            model.load('../models/resnet_svhn.pth')
    else:
        raise ValueError(f'{model_name} is not supported')
    # add_feature_extractor_method(model)
    model.eval()
    model = model.cuda() if torch.cuda.is_available() else model
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
                                 dataloader: data.DataLoader, is_dev_run: bool = False) -> (list, list, list, list):
    features_list, labels_list, outputs_list, prob_list = [], [], [], []
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
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

            if is_dev_run is True:
                break

    return features_list, labels_list, outputs_list, prob_list


def extract_features(model: torch.nn.Module, dataloader, odin_eps: float = 0.0, is_dev_run: bool = False):
    if odin_eps == 0.0:
        features, labels, outputs, probs = extract_features_from_loader(model, dataloader, is_dev_run=is_dev_run)
    else:
        features, labels, outputs, probs = odin_extract_features_from_loader(model, dataloader, odin_eps,
                                                                             is_dev_run=is_dev_run)

    features_dataset = FeaturesDataset(features, labels, outputs, probs,
                                       transform=transforms.Compose([transforms.Lambda(lambda x: x)]))
    torch.cuda.empty_cache()
    return features_dataset
