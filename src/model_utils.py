import logging
import sys
import warnings
from glob import glob
import os
import torch
from torch import nn
from torch.serialization import SourceChangeWarning
import torchvision.models as models
from model_arch_utils.densenet import DenseNet3
from model_arch_utils.densenet_gram import DenseNet3Gram
from model_arch_utils.resnet import ResNet34
from model_arch_utils.resnet_gram import ResNet34Gram
from model_arch_utils.wrn import WideResNet
import types

sys.path.append("./model_arch_utils")

warnings.filterwarnings("ignore", category=SourceChangeWarning)
logger = logging.getLogger(__name__)


def get_model(
    model_name: str, trainset_name: str, is_pretrained: bool = True
) -> torch.nn.Module:
    if model_name == "densenet":
        if trainset_name in ["cifar10", "svhn"]:
            model = DenseNet3(100, 10)
        elif trainset_name == "cifar100":
            model = DenseNet3(100, 100)
        else:
            raise ValueError(f"trainset_name={trainset_name} is not supported")
    elif model_name == "resnet":
        if trainset_name in ["cifar10", "svhn"]:
            model = ResNet34(num_c=10)
        elif trainset_name == "cifar100":
            model = ResNet34(num_c=100)
        else:
            raise ValueError(f"trainset_name={trainset_name} is not supported")
    elif model_name.endswith("imagenet"):
        model = get_imagenet_pretrained_resnet(model_name)
    else:
        raise ValueError(f"model_name={model_name} is not supported")

    # Load pretrained weights
    if is_pretrained is True and not model_name.endswith("imagnet"):
        path = f"../models/{model_name}_{trainset_name}.pth"
        logger.info(f"Load pretrained model: {path}")
        model.load(path)
    return model


def get_gram_model(
    model_name: str, trainset_name: str, is_pretrained: bool = True
) -> torch.nn.Module:
    if model_name == "densenet":
        if trainset_name in ["cifar10", "svhn"]:
            model = DenseNet3Gram(100, 10)
        elif trainset_name == "cifar100":
            model = DenseNet3Gram(100, 100)
        else:
            raise ValueError(f"trainset_name={trainset_name} is not supported")
    elif model_name == "resnet":
        if trainset_name in ["cifar10", "svhn"]:
            model = ResNet34Gram(num_c=10)
        elif trainset_name == "cifar100":
            model = ResNet34Gram(num_c=100)
        else:
            raise ValueError(f"trainset_name={trainset_name} is not supported")
    else:
        raise ValueError(f"model_name={model_name} is not supported")

    if is_pretrained is True:
        path = f"../models/{model_name}_{trainset_name}.pth"
        logger.info(f"Load pretrained model: {path}. {os.getcwd()}")
        model.load(path)
    return model


def get_energy_model(
    model_name: str, trainset_name: str, is_pretrained: bool = True
) -> torch.nn.Module:
    assert model_name == "wrn", "Only wrn model is supported"
    assert trainset_name in ["cifar10", "cifar100"], "Only cifar10 set is supported"

    if model_name == "wrn":
        if trainset_name == "cifar10":
            model = WideResNet(depth=40, num_classes=10, widen_factor=2)
        elif trainset_name == "cifar100":
            model = WideResNet(depth=40, num_classes=100, widen_factor=2)
        else:
            raise ValueError(f"trainset_name={trainset_name} is not supported")
    else:
        raise ValueError(f"model_name={model_name} is not supported")

    if is_pretrained is True:
        path = f"../models/{trainset_name}_{model_name}_s1_energy_ft_epoch_9.pt"
        logger.info(f"Load pretrained model: {path}")
        model.load_state_dict(torch.load(path))
    return model


def get_oecc_model(
    model_name: str, trainset_name: str, is_pretrained: bool = True
) -> torch.nn.Module:
    model = get_gram_model(model_name, trainset_name, is_pretrained=False)

    if is_pretrained is True:
        model_name = "ResNet34" if model_name == "resnet" else model_name
        path = glob(
            f"../models/Zero_Shot_{trainset_name}_{model_name}_OECC_tune_epoch_*.pt*"
        )[0]
        logger.info(f"Load pretrained model: {path}")
        model.load(path)
    return model


def get_imagenet_pretrained_resnet(model_name: str):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        self.features_out = x.clone()
        x = self.fc(x)
        return x

    def get_features(self):
        """
        Added this method for pNML ood detection
        :return:
        """
        return self.features_out

    if model_name == "resnet18_imagenet":
        model = models.resnet18(pretrained=True)
    elif model_name == "resnet101_imagenet":
        model = models.resnet101(pretrained=True)
    else:
        raise ValueError(f"{model_name=} is not supported")
    model.forward = types.MethodType(forward, model)
    model.get_features = types.MethodType(get_features, model)
    return model
