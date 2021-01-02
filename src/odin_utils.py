# from pytorchcv.model_provider import get_model as ptcv_get_model
import logging
import os
import os.path as osp

import torch
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

from dataset_utils import FeaturesDataset
from save_product_utils import save_products

logger = logging.getLogger(__name__)


def extract_odin_features(model, loaders_dict: dict, model_name: str, trainset_name: str, out_dir: str, odin_dict: dict,
                          is_dev_run: bool = False):
    data_name = 'trainset'
    test_loader = loaders_dict.pop(data_name)
    logger.info('Feature extraction for {}'.format(data_name))
    features_dataset = odin_extract_features_from_loader(model, test_loader, eps=0.0, is_dev_run=is_dev_run)
    save_products(features_dataset, out_dir, data_name)
    logger.info('')

    # Get ind testset
    ind_loader = loaders_dict.pop(trainset_name)
    ind_name = trainset_name

    # Extract features for all dataset: trainset, ind_testset and ood_testsets
    for data_name, ood_loader in loaders_dict.items():
        # Get ODIN parameters if available
        odin_eps = odin_dict[model_name][trainset_name][data_name]
        testset_out_dir = osp.join(out_dir, data_name)
        os.makedirs(testset_out_dir)

        # Extract features
        logger.info('Feature extraction for {}. odin_eps={}'.format(data_name, odin_eps))

        # ind
        features_dataset = odin_extract_features_from_loader(model, ind_loader, odin_eps, is_dev_run=is_dev_run)
        save_products(features_dataset, testset_out_dir, ind_name)

        # ood
        features_dataset = odin_extract_features_from_loader(model, ood_loader, odin_eps, is_dev_run=is_dev_run)
        save_products(features_dataset, testset_out_dir, data_name)
        logger.info('')


def perturbate_input(model, images, magnitude: float, temper: float = 1000):
    """
    Execute adversarial attack on the input image.
    :param model: pytorch model to use.
    :param images: image to attack.
    :param magnitude: the attack strength
    :param temper: smoothing factor of the logits.
    :return: attacked image
    """
    criterion = torch.nn.CrossEntropyLoss()
    model.zero_grad()

    # Forward
    images = images.cuda() if torch.cuda.is_available() else images
    images.requires_grad = True
    outputs = model(images)

    # Using temperature scaling
    outputs = outputs / temper

    # Calculating the perturbation we need to add, that is,
    # the sign of gradient of cross entropy loss w.r.t. input
    pseudo_labels = torch.argmax(outputs, dim=1).detach()
    loss = criterion(outputs, pseudo_labels)
    loss.backward()

    # Normalizing the gradient to binary in {0, 1}
    gradient = torch.ge(images.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2

    # Adding small perturbations to images
    imgs_perturbs = torch.add(images.data, -gradient, alpha=magnitude)
    imgs_perturbs.requires_grad = False
    model.zero_grad()

    return imgs_perturbs


def odin_extract_features_from_loader(model: torch.nn.Module, dataloader: data.DataLoader,
                                      eps: float, temper: float = 1000,
                                      num_samples=None,
                                      is_dev_run: bool = False) -> FeaturesDataset:
    features_list, labels_list, outputs_list, prob_list = [], [], [], []
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()

    for batch_n, (images, labels) in enumerate(tqdm(dataloader)):
        images = images.to(device)
        labels = labels.to(device)

        imgs_perturbs = perturbate_input(model, images, eps, temper)

        # Forward pass
        outputs = model(imgs_perturbs)
        outputs = outputs / temper
        probs = torch.nn.functional.softmax(outputs, dim=-1)

        # Get Features
        features_list.append(model.get_features().cpu().detach())
        labels_list.append(labels.cpu())
        outputs_list.append(outputs.cpu().detach())
        prob_list.append(probs.cpu().detach())

        if is_dev_run is True and batch_n > 1:
            break

        if num_samples is not None and batch_n * len(images) > num_samples:
            break

    features_dataset = FeaturesDataset(features_list, labels_list, outputs_list, prob_list,
                                       transform=transforms.Compose([transforms.Lambda(lambda x: x)]))
    torch.cuda.empty_cache()
    return features_dataset
