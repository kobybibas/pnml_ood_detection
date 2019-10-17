import os.path as osp

import numpy as np
import torch
from loguru import logger
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from tqdm import tqdm

from dataset_classes import FeaturesDataset, UniformNoiseDataset, GaussianNoiseDataset, ImageFolderOOD

dataset_class_valid = (CIFAR10, CIFAR100, UniformNoiseDataset, GaussianNoiseDataset, ImageFolderOOD)


def extract_features_from_dataloader(model: torch.nn.Module, dataloader: data.DataLoader) -> (list, list, list):
    features_list = []
    labels_list = []
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    outputs_list = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            # Forward pass
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            # Get Features
            features_list.append(model.get_features().cpu().detach())
            labels_list.append(labels.cpu())
            outputs_list.append(outputs.cpu().detach())

    return features_list, labels_list, outputs_list


def extract_features(model: torch.nn.Module, dataloader):
    # Extract features and outputs from dataloader
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()
    features_list, labels_list, outputs_list = extract_features_from_dataloader(model, dataloader)

    features_dataset = FeaturesDataset(features_list,
                                       labels_list,
                                       outputs_list,
                                       transform=transforms.Compose([
                                           transforms.Lambda(lambda x: x)]))
    torch.cuda.empty_cache()
    return features_dataset


def save_train_labels(trainset: dataset_class_valid, trainset_name: str, output_folder: str):
    file_name = osp.join(output_folder, '{}_train_labels.npy'.format(trainset_name))
    logger.info('Saving to {}'.format(file_name))
    np.save(file_name, trainset.targets)


def save_products(features_dataset: FeaturesDataset,
                  features_dir: str,
                  logits_dir: str,
                  model_name: str,
                  trainset_name: str,
                  set_name: str,
                  set_type: str):
    """
    Save dataset features and outputs logits.
    :param features_dataset:
    :param output_folder:
    :param model_name:
    :param trainset_name:
    :param set_name:
    :param set_type:
    :return:
    """
    prefix = '{}_{}_{}_{}'.format(model_name, trainset_name, set_name, set_type)

    # Save Features
    file_name = osp.join(features_dir, '{}_features.npy'.format(prefix))
    logger.info('Features shape {}. Saving to {}'.format(features_dataset.data.shape, file_name))
    np.save(file_name, features_dataset.data)

    # Save Outputs
    file_name = osp.join(logits_dir, '{}_outputs.npy'.format(prefix))
    logger.info('Outputs shape {}. Saving to {}'.format(features_dataset.outputs.shape, file_name))
    np.save(file_name, features_dataset.outputs)
