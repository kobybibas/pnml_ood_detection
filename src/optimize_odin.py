import itertools
import json
import logging
import os
import os.path as osp

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from dataset_utils import get_dataloaders
from model_utils import get_model, test_pretrained_model, extract_features_from_loader
from odin_utils import odin_extract_features_from_loader
from score_utils import calc_metrics_transformed as calc_metrics
from score_utils import transform_features, calc_regret_on_set, calc_projection_matrices, add_bias_term

logger = logging.getLogger(__name__)

"""
Example of running:
CUDA_VISIBLE_DEVICES=0 python optimize_odin.py -model densenet -trainset cifar10 -num_workers 0
"""


def optimize_odin_to_dataset(model, ind_loader, ood_loader, trainloader,
                             num_samples: int,
                             epsilons: list, temperatures: list, model_name: str,
                             is_with_pnml: bool,
                             is_dev_run: bool = False):
    p_parallel, p_bot = 0, 0
    if is_with_pnml is True:
        logger.info('extract_features_from_loader: trainset')
        features_dataset = extract_features_from_loader(model, trainloader, is_dev_run=is_dev_run)
        trainset_features = features_dataset.data
        trainset_features = transform_features(add_bias_term(trainset_features))
        p_parallel, p_bot = calc_projection_matrices(trainset_features)

    best_tnr, best_eps, best_temperature = 0.0, 0.0, 1.0
    for i, (temperature, eps) in enumerate(itertools.product(temperatures, epsilons)):
        features_dataset = odin_extract_features_from_loader(model, ind_loader, eps, temperature, model_name,
                                                             num_samples=num_samples, is_dev_run=is_dev_run)
        probs_ind = features_dataset.probs
        features_ind = features_dataset.data
        features_dataset = odin_extract_features_from_loader(model, ood_loader, eps, temperature, model_name,
                                                             num_samples=num_samples, is_dev_run=is_dev_run)
        probs_ood = features_dataset.probs
        features_ood = features_dataset.data

        if is_with_pnml is False:
            max_prob_ind = probs_ind.max(axis=1)
            max_prob_ood = probs_ood.max(axis=1)

            labels = [1] * len(max_prob_ind) + [0] * len(max_prob_ood)
            scores = np.append(max_prob_ind, max_prob_ood)
            performance_dict = calc_metrics(scores, labels)
            tnr = performance_dict['TNR at TPR 95%']
        else:
            features_ind = transform_features(add_bias_term(features_ind))
            regrets_ind, _ = calc_regret_on_set(features_ind, probs_ind, p_parallel, p_bot)
            features_ood = transform_features(add_bias_term(features_ood))
            regrets_ood, _ = calc_regret_on_set(features_ood, probs_ood, p_parallel, p_bot)

            labels = [1] * len(regrets_ind) + [0] * len(regrets_ood)
            scores = -np.append(regrets_ind, regrets_ood)
            performance_dict = calc_metrics(scores, labels)
            tnr = performance_dict['TNR at TPR 95%']

        logger.info('[eps temperature tnr]=[{} {} {:.3f}]'.format(eps, temperature, tnr))
        if tnr > best_tnr:
            logger.info('    New best tnr.')
            best_eps = eps
            best_temperature = temperature
            best_tnr = tnr

        if is_dev_run:
            break
    return best_eps, best_temperature, best_tnr


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
    trainloader = loaders_dict.pop('trainset')
    ind_loader = loaders_dict.pop(cfg.trainset)

    # Optimize odin_pnml for each dataset
    best_eps_dict = {}
    for i, (data_name, ood_loader) in enumerate(loaders_dict.items()):
        epsilon, temperature, tnr = optimize_odin_to_dataset(model, ind_loader, ood_loader, trainloader,
                                                             cfg.num_samples,
                                                             cfg.epsilons, cfg.temperatures, cfg.model,
                                                             cfg.is_with_pnml,
                                                             is_dev_run=cfg.dev_run)
        logger.info('[{}/{}] {}: best [epsilon temperature tnr]=[{} {} {}]'.format(i, len(loaders_dict.keys()) - 1,
                                                                                   data_name,
                                                                                   epsilon, temperature, tnr))
        best_eps_dict[data_name] = {'epsilon': epsilon,
                                    'temperature': temperature}
        if cfg.dev_run:
            break

    with open(osp.join(out_dir, 'optimized_odin_params.json'), 'w') as f:
        json.dump(best_eps_dict, f, ensure_ascii=True, indent=4, sort_keys=True)

    logger.info('Finished!')


if __name__ == "__main__":
    optimize_odin()
