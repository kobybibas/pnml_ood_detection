# from pytorchcv.model_provider import get_model as ptcv_get_model
import logging
import os
import os.path as osp

import numpy as np
import torch
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

from dataset_utils import FeaturesDataset
from save_product_utils import save_products
from score_utils import calc_metrics_transformed as calc_metrics

logger = logging.getLogger(__name__)


def extract_odin_features(model, loaders_dict: dict, model_name: str, trainset_name: str, out_dir: str,
                          odin_dict: dict, odin_pnml_dict: dict,
                          num_skip_samples: int = 0,
                          is_dev_run: bool = False):
    data_name = 'trainset'
    test_loader = loaders_dict.pop(data_name)
    logger.info('Feature extraction for {}'.format(data_name))
    features_dataset = odin_extract_features_from_loader(model, test_loader, epsilon=0.0, temperature=1.0,
                                                         num_skip_samples=0,
                                                         model_name=model_name, is_dev_run=is_dev_run)
    save_products(features_dataset, out_dir, data_name)
    logger.info('')

    # Get ind testset
    ind_loader = loaders_dict.pop(trainset_name)
    ind_name = trainset_name

    # Extract features for all dataset: trainset, ind_testset and ood_testsets
    for data_name, ood_loader in loaders_dict.items():

        # Get ODIN parameters
        epsilon_odin = odin_dict[f'{model_name}_{trainset_name}'][data_name]['epsilon']
        temperature_odin = odin_dict[f'{model_name}_{trainset_name}'][data_name]['temperature']
        testset_out_dir_odin = osp.join(out_dir, data_name)

        # Get ODIN+pNML parameters
        epsilon_pnml = odin_pnml_dict[f'{model_name}_{trainset_name}'][data_name]['epsilon']
        temperature_pnml = odin_pnml_dict[f'{model_name}_{trainset_name}'][data_name]['temperature']
        testset_out_dir_pnml = osp.join(out_dir, data_name + '_pNML')

        for epsilon, temperature, testset_out_dir, method in (
                (epsilon_odin, temperature_odin, testset_out_dir_odin, 'ODIN'),
                (epsilon_pnml, temperature_pnml, testset_out_dir_pnml, 'ODIN+pNML')):
            os.makedirs(testset_out_dir)

            # Extract features
            logger.info('{} for {}. [epsilon temperature]=[{} {}]'.format(method, data_name, epsilon, temperature))

            # ind
            features_dataset = odin_extract_features_from_loader(model, ind_loader, epsilon, temperature, model_name,
                                                                 num_skip_samples=num_skip_samples,
                                                                 is_dev_run=is_dev_run)
            save_products(features_dataset, testset_out_dir, ind_name)
            probs_ind = features_dataset.probs

            # ood
            features_dataset = odin_extract_features_from_loader(model, ood_loader, epsilon, temperature, model_name,
                                                                 num_skip_samples=num_skip_samples,
                                                                 is_dev_run=is_dev_run)
            save_products(features_dataset, testset_out_dir, data_name)
            probs_ood = features_dataset.probs

            # score
            max_prob_ind = probs_ind.max(axis=1)
            max_prob_ood = probs_ood.max(axis=1)

            if method == 'ODIN':
                labels = [1] * len(max_prob_ind) + [0] * len(max_prob_ood)
                scores = np.append(max_prob_ind, max_prob_ood)
                performance_dict = calc_metrics(scores, labels)
                logger.info(performance_dict)
            logger.info('')


def perturbate_input(model, images, epsilon: float, temperature: float, model_name: str):
    """
    Execute adversarial attack on the input image.
    :param model: pytorch model to use.
    :param images: image to attack.
    :param epsilon: the attack strength
    :param temperature: smoothing factor of the logits.
    :param model_name: name of architecture
    :return: attacked image
    """
    criterion = torch.nn.CrossEntropyLoss()
    model.zero_grad()

    # Forward
    images = images.cuda() if torch.cuda.is_available() else images
    images.requires_grad = True
    outputs = model(images)

    # Using temperature scaling
    outputs = outputs / temperature

    # Calculating the perturbation we need to add, that is,
    # the sign of gradient of cross entropy loss w.r.t. input
    pseudo_labels = torch.argmax(outputs, dim=1).detach()
    loss = criterion(outputs, pseudo_labels)
    loss.backward()

    # Normalizing the gradient to binary in {0, 1}
    gradient = torch.ge(images.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2

    if model_name == 'densenet':
        gradient.index_copy_(1, torch.LongTensor([0]).cuda(),
                             gradient.index_select(1, torch.LongTensor([0]).cuda()) / (63.0 / 255.0))
        gradient.index_copy_(1, torch.LongTensor([1]).cuda(),
                             gradient.index_select(1, torch.LongTensor([1]).cuda()) / (62.1 / 255.0))
        gradient.index_copy_(1, torch.LongTensor([2]).cuda(),
                             gradient.index_select(1, torch.LongTensor([2]).cuda()) / (66.7 / 255.0))
    elif model_name == 'resnet':
        gradient.index_copy_(1, torch.LongTensor([0]).cuda(),
                             gradient.index_select(1, torch.LongTensor([0]).cuda()) / (0.2023))
        gradient.index_copy_(1, torch.LongTensor([1]).cuda(),
                             gradient.index_select(1, torch.LongTensor([1]).cuda()) / (0.1994))
        gradient.index_copy_(1, torch.LongTensor([2]).cuda(),
                             gradient.index_select(1, torch.LongTensor([2]).cuda()) / (0.2010))
    else:
        raise ValueError(f'{model_name} is not supported')

    # Adding small perturbations to images
    imgs_perturbs = torch.add(images.data, -gradient, alpha=epsilon)
    imgs_perturbs.requires_grad = False
    model.zero_grad()

    return imgs_perturbs


def odin_extract_features_from_loader(model: torch.nn.Module, dataloader: data.DataLoader,
                                      epsilon: float, temperature: float, model_name: str,
                                      num_samples=None, num_skip_samples: int = 0,
                                      is_dev_run: bool = False) -> FeaturesDataset:
    features_list, labels_list, outputs_list, prob_list = [], [], [], []
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()

    total = 0
    for batch_n, (images, labels) in enumerate(tqdm(dataloader)):
        images = images.to(device)
        labels = labels.to(device)
        total += images.size(0)

        if total <= num_skip_samples:
            continue

        if epsilon == 0.0:
            imgs_perturbs = images
        else:
            imgs_perturbs = perturbate_input(model, images, epsilon, temperature, model_name)

        # Forward pass
        with torch.no_grad():
            outputs = model(imgs_perturbs)
        outputs = outputs / temperature
        probs = torch.nn.functional.softmax(outputs, dim=-1)

        # Get Features
        features_list.append(model.get_features().cpu().detach())
        labels_list.append(labels.cpu())
        outputs_list.append(outputs.cpu().detach())
        prob_list.append(probs.cpu().detach())

        if is_dev_run is True and batch_n > 1:
            break

        if num_samples is not None and total >= num_samples:
            break

    features_dataset = FeaturesDataset(features_list, labels_list, outputs_list, prob_list,
                                       transform=transforms.Compose([transforms.Lambda(lambda x: x)]))
    torch.cuda.empty_cache()
    return features_dataset
