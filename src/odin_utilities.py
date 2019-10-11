import time
from os import path as osp

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

from dataset_classes import FeaturesDataset
from experimnet_utilities import Experiment
from result_tracker_utilities import ResultTracker


def perturbate_dataset(model: torch.nn.Module, dataloader: data.DataLoader, noise_magnitude: float, temper: float):
    norm = [63.0 / 255.0, 62.1 / 255.0, 66.7 / 255.0]
    features_list, labels_list = [], []
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    time_start = time.time()
    model.eval()
    model.to(device)
    for images, labels_org in tqdm(dataloader):
        images = images.to(device)
        images.requires_grad = True
        outputs = model(images)

        # Using temperature scaling
        outputs = outputs / temper

        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        labels = torch.argmax(outputs, axis=1)
        loss = criterion(outputs, labels)
        model.zero_grad()
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(images.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        # Normalizing the gradient to the same space of image
        gradient[:, 0] = (gradient[:, 0]) / (norm[0])
        gradient[:, 1] = (gradient[:, 1]) / (norm[1])
        gradient[:, 2] = (gradient[:, 2]) / (norm[2])
        # Adding small perturbations to images

        inputs_temp = torch.add(images.data, -noise_magnitude, gradient)

        features_list.append(inputs_temp.cpu())
        labels_list.append(labels_org.cpu())

    torch.cuda.empty_cache()
    dataset_perturbed = FeaturesDataset(features_list,
                                        labels_list,
                                        transform=transforms.Compose([
                                            transforms.Lambda(lambda x: x)]))
    logger.debug('Created dataset_perturbed in {:.2f} sec'.format(time.time() - time_start))

    # Create dataloader
    testloader_perturbed = data.DataLoader(dataset_perturbed,
                                           batch_size=dataloader.batch_size,
                                           num_workers=dataloader.num_workers)
    return testloader_perturbed


def execute_odin_baseline(model: torch.nn.Module, dataloader: data.DataLoader, temper: float,
                          experiment_h: Experiment, tracker: ResultTracker):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    time_start = time.time()
    model.eval()
    model.to(device)

    max_prob_list = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images, labels = images.to(device), images.to(device)

            outputs = model(images)
            max_output, _ = torch.max(outputs, axis=1, keepdim=True)
            outputs -= max_output
            prob = F.softmax((outputs - max_output) / temper, dim=-1)
            max_prob, _ = torch.max(prob, axis=1)
            max_prob_list.append(max_prob.detach().cpu())
        torch.cuda.empty_cache()
        logger.info('Finish calculate max prob in {:.2f} sec'.format(time.time() - time_start))
    max_prob_np = torch.cat(max_prob_list).cpu().numpy()
    file_name = osp.join(tracker.output_dir,
                         '{}_{}_test_odin.npy'.format(experiment_h.exp_type, experiment_h.testset_name))
    logger.info('Saving to {}'.format(file_name))
    np.save(file_name, max_prob_np)
