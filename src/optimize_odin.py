import json
import logging
import os
import os.path as osp

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from ood_metrics import calc_metrics

from dataset_utils import get_dataloaders
from model_utils import get_model, test_pretrained_model
from odin_utils import odin_extract_features_from_loader

logger = logging.getLogger(__name__)

"""
Example of running:
CUDA_VISIBLE_DEVICES=0 python optimize_odin.py -model densenet -trainset cifar10 -num_workers 0
"""


def optimize_odin_fo_dataset(model, ind_loader, ood_loader,
                             num_samples,
                             eps_min, eps_max, eps_num, eps_levels,
                             is_dev_run: bool = False):
    best_auroc, best_eps = 0.0, 0.0
    for eps_level in range(eps_levels):
        epsilon_list = np.linspace(eps_min, eps_max, eps_num)
        best_i = np.argmin(np.abs(best_eps - epsilon_list))

        logger.info('eps level [{}/{}]: epsilon_list={}'.format(eps_level, eps_levels - 1, epsilon_list))
        for i, eps in enumerate(epsilon_list):
            features_dataset = odin_extract_features_from_loader(model, ind_loader, eps,
                                                                 num_samples=num_samples,
                                                                 is_dev_run=is_dev_run)
            prob_ind = features_dataset.probs
            features_dataset = odin_extract_features_from_loader(model, ood_loader, eps,
                                                                 num_samples=num_samples,
                                                                 is_dev_run=is_dev_run)
            prob_ood = features_dataset.probs

            max_prob_ind = prob_ind.max(axis=1)
            max_prob_ood = prob_ood.max(axis=1)

            labels = [1] * len(max_prob_ind) + [0] * len(max_prob_ood)
            scores = np.append(max_prob_ind, max_prob_ood)
            performance_dict = calc_metrics(scores, labels)
            auroc = performance_dict['auroc']
            logger.info('[eps auroc]=[{} {:.3f}]'.format(eps, auroc))
            if auroc > best_auroc:
                logger.info('    New best auroc.')
                best_eps = eps
                best_auroc = auroc
                best_i = i

        if best_i == 0 or best_i == len(epsilon_list) - 1:
            logger.warning('Warning. best_i is at the edge: {}'.format(best_i))
        eps_min, eps_max = epsilon_list[max(0, best_i - 1)], epsilon_list[min(len(epsilon_list) - 1, best_i + 1)]

        if is_dev_run:
            break
    return best_eps, best_auroc


@hydra.main(config_path="../configs", config_name="optimize_odin")
def optimize_odin(cfg: DictConfig):
    logger.info(cfg)
    out_dir = os.getcwd()
    os.chdir(hydra.utils.get_original_cwd())
    logger.info('torch.cuda.is_available={}'.format(torch.cuda.is_available()))

    # Load pretrained model
    logger.info('Get Model: {} {}'.format(cfg.model, cfg.trainset))
    model = get_model(cfg.model, cfg.trainset)

    # Load datasets
    logger.info('Load datasets: {}'.format(cfg.data_dir))
    loaders_dict = get_dataloaders(cfg.model,
                                   cfg.trainset, cfg.data_dir, cfg.batch_size,
                                   cfg.num_workers if cfg.dev_run is False else 0)
    assert 'trainset' in loaders_dict  # Contains the trainset loader
    assert cfg.trainset in loaders_dict  # This is the in-distribution testset loader

    logger.info('Datasets: {}'.format(loaders_dict.keys()))
    if cfg.test_pretrained is True:
        logger.info('Testing pretrained model')
        ind_testset = loaders_dict[cfg.trainset]
        test_pretrained_model(model, loaders_dict['trainset'], ind_testset, is_dev_run=cfg.dev_run)

    # Remove trainset
    loaders_dict.pop('trainset')
    ind_loader = loaders_dict.pop(cfg.trainset)

    # Optimize odin for each dataset
    best_eps_dict = {}
    for data_name, ood_loader in loaders_dict.items():
        logger.info('Optimize odin for {}'.format(data_name))
        epsilon, auroc = optimize_odin_fo_dataset(model, ind_loader, ood_loader,
                                                  cfg.num_samples,
                                                  cfg.eps_min, cfg.eps_max, cfg.eps_num, cfg.eps_levels,
                                                  is_dev_run=cfg.dev_run)
        logger.info('{}: best [epsilon auroc]=[{} {}]'.format(data_name, epsilon, auroc))
        best_eps_dict[data_name] = epsilon

        if cfg.dev_run:
            break

    with open(osp.join(out_dir, 'optimized_eps.json'), 'w') as f:
        json.dump(best_eps_dict, f, ensure_ascii=True, indent=4, sort_keys=True)

    logger.info('Finished!')


if __name__ == "__main__":
    optimize_odin()
