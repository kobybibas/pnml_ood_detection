import logging
import os
import os.path as osp

import hydra
import torch
from omegaconf import DictConfig

from dataset_utils import get_dataloaders
from model_utils import get_model, test_pretrained_model, extract_features
from save_product_utils import save_last_layer_weights, save_products, save_train_labels

logger = logging.getLogger(__name__)

"""
Example of running:
CUDA_VISIBLE_DEVICES=0 python main.py -model densenet -trainset cifar10 -num_workers 0
"""


def extract_odin_features(model, loaders_dict: dict, model_name: str, trainset_name: str, out_dir: str, odin_dict: dict,
                          is_dev_run: bool = False):
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
        features_dataset = extract_features(model, ind_loader, odin_eps, is_dev_run=is_dev_run)
        save_products(features_dataset, testset_out_dir, model_name, trainset_name, ind_name)

        # ood
        features_dataset = extract_features(model, ood_loader, odin_eps, is_dev_run=is_dev_run)
        save_products(features_dataset, testset_out_dir, model_name, trainset_name, data_name)
        logger.info('')


def extract_baseline_features(model, loaders_dict: dict, model_name, trainset_name: str, out_dir: str,
                              is_dev_run: bool = False):
    # Extract features for all dataset: trainset, ind_testset and ood_testsets
    for data_name, loader in loaders_dict.items():
        logger.info('Feature extraction for {}'.format(data_name))
        features_dataset = extract_features(model, loader, odin_eps=0.0, is_dev_run=is_dev_run)
        save_products(features_dataset, out_dir, model_name, trainset_name, data_name)
        logger.info('')


# @hydra.main(config_path="../configs", config_name="extract_odin.yaml")
@hydra.main(config_path="../configs", config_name="extract_baseline")
def run_experiment(cfg: DictConfig):
    logger.info(cfg)
    out_dir = os.getcwd()
    os.chdir(hydra.utils.get_original_cwd())
    logger.info('torch.cuda.is_available={}'.format(torch.cuda.is_available()))

    # Load pretrained model
    logger.info('Get Model: {} {}'.format(cfg.model, cfg.trainset))
    model = get_model(cfg.model, cfg.trainset)
    save_last_layer_weights(model, out_dir)

    # Load datasets
    logger.info('Load datasets: {}'.format(cfg.data_dir))
    loaders_dict = get_dataloaders(cfg.model,
                                   cfg.trainset, cfg.data_dir, cfg.batch_size,
                                   cfg.num_workers if cfg.dev_run is False else 0)
    assert 'trainset' in loaders_dict  # Contains the trainset loader
    assert cfg.trainset in loaders_dict  # This is the in-distribution testset loader

    # Save labels
    logger.info('Save labels for {}'.format(cfg.trainset))
    save_train_labels(loaders_dict['trainset'].dataset, cfg.trainset, out_dir)

    logger.info('Datasets: {}'.format(loaders_dict.keys()))
    if cfg.test_pretrained is True:
        logger.info('Testing pretrained model')
        ind_testset = loaders_dict[cfg.trainset]
        test_pretrained_model(model, loaders_dict['trainset'], ind_testset, is_dev_run=cfg.dev_run)

    # Extract trainset features:
    data_name = 'trainset'
    loader = loaders_dict.pop(data_name)
    logger.info('Feature extraction for {}'.format(data_name))
    features_dataset = extract_features(model, loader, odin_eps=0.0, is_dev_run=cfg.dev_run)
    save_products(features_dataset, out_dir, cfg.model, cfg.trainset, data_name)
    logger.info('')

    # Extract features
    if not hasattr(cfg, 'odin'):
        extract_baseline_features(model, loaders_dict, cfg.model, cfg.trainset, out_dir, is_dev_run=cfg.dev_run)
    else:
        extract_odin_features(model, loaders_dict, cfg.model, cfg.trainset, out_dir, cfg.odin, is_dev_run=cfg.dev_run)
    logger.info('Finished!')


if __name__ == "__main__":
    run_experiment()
