import os.path as osp
import types

import torch
from loguru import logger
from pytorchcv.model_provider import get_model as ptcv_get_model
from torch import nn
from torch.utils import data
from tqdm import tqdm


def add_feature_extractor_method(model: torch.nn.Module):
    # Add feature extractor method
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


def get_model(model_path: str, model_name: str, trainset_name: str) -> torch.nn.Module:
    # Get list of pretrained models

    model = None
    if model_name == 'densenet':
        model = ptcv_get_model("densenet100_k12_bc_%s" % trainset_name, pretrained=False)
    elif model_name == 'resnet':
        model = ptcv_get_model("wrn28_10_%s" % trainset_name, pretrained=False)
    assert isinstance(model, torch.nn.Module)

    add_feature_extractor_method(model)
    model_path = osp.join(model_path, '{}_{}.pth'.format(model_name, trainset_name))
    logger.info('Loading pretrained from: {}'.format(model_path))
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def test_pretrained_model(model: nn.Module, trainloader: data.DataLoader, testloader: data.DataLoader):
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

            acc = correct / len(dataloader.dataset)
            loss /= len(dataloader.dataset)
            logger.info(
                'Pretrained model: {} [Acc Error Loss]=[{:.2f}% {:.2f}% {:.3f}]'.format(data_type,
                                                                                        100 * acc,
                                                                                        100 - 100 * acc,
                                                                                        loss))
