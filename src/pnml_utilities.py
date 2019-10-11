import os.path as osp
import time

import numpy as np
import torch
from loguru import logger
from torchvision import transforms
from tqdm import tqdm

from dataset_classes import FeaturesDataset
from experimnet_utilities import Experiment


def extract_features_from_dataloader(model, dataloader, is_in_distribution: bool = False):
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    features_list = []
    labels_list = []
    loss_sum = 0
    correct_sum = 0
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    outputs_list = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            # In case testing in distribution data
            if is_in_distribution is True:
                loss = criterion(outputs, labels)
                loss_sum += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct_sum += (predicted == labels).sum().item()

            # Get Features
            features_list.append(model.get_features().cpu())
            labels_list.append(labels.cpu())
            outputs_list.append(outputs.cpu().detach())

        acc = correct_sum / len(dataloader.dataset)
        loss = loss_sum / len(dataloader.dataset)
    return features_list, labels_list, outputs_list, acc, loss


def extract_features(model: torch.nn.Module, dataloaders: dict, experiment_h: Experiment):
    # Extract features from train and test
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()
    features_datasets_dict = {}
    is_in_distribution = experiment_h.trainset_name == experiment_h.testset_name
    set_type_list = ['train', 'test'] if is_in_distribution else ['test']
    for set_type in set_type_list:
        time_start = time.time()
        features_list, labels_list, outputs_list, acc, loss = extract_features_from_dataloader(model,
                                                                                               dataloaders[set_type],
                                                                                               is_in_distribution)

        features_datasets_dict[set_type] = FeaturesDataset(features_list,
                                                           labels_list,
                                                           outputs_list,
                                                           transform=transforms.Compose([
                                                               transforms.Lambda(lambda x: x)]))
        torch.cuda.empty_cache()
        logger.debug('Feature extraction {}. [Acc Loss]=[{:.2f}% {:.2f}] in {:.2f} sec'.format(set_type,
                                                                                               100 * acc,
                                                                                               loss,
                                                                                               time.time() - time_start))
        torch.cuda.empty_cache()
    return features_datasets_dict


def save_features(features_datasets_dict: dict, experiment_h: Experiment, output_folder: str,
                  suffix: str = ''):
    # Save features
    exp_type = experiment_h.exp_type
    for set_type, feature_dataset in features_datasets_dict.items():
        set_name = experiment_h.trainset_name if set_type == 'train' else experiment_h.testset_name
        file_name = osp.join(output_folder,
                             '{}_{}_{}_pnml{}.npy'.format(exp_type, set_name, set_type, suffix))
        logger.info('Saving to {}. dataset shape {}'.format(file_name, feature_dataset.data.shape))
        np.save(file_name, feature_dataset.data)
        file_name = osp.join(output_folder,
                             '{}_{}_{}_outputs_pnml{}.npy'.format(exp_type, set_name, set_type, suffix))
        logger.info('Saving to {}. dataset shape {}'.format(file_name, feature_dataset.outputs.shape))
        np.save(file_name, feature_dataset.outputs)
    if 'train' in features_datasets_dict.keys():
        file_name = osp.join(output_folder, '{}_train_labels.npy'.format(experiment_h.trainset_name))
        logger.info('Saving to {}'.format(file_name))
        np.save(file_name, features_datasets_dict['train'].targets)
