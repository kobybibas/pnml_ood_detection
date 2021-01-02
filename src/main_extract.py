import logging
import os

import hydra
import torch
from omegaconf import DictConfig

from dataset_utils import get_dataloaders
from gram_utils import extract_gram_features
from model_utils import get_model, test_pretrained_model, extract_baseline_features
from odin_utils import extract_odin_features
from save_product_utils import save_last_layer_weights, save_train_labels

logger = logging.getLogger(__name__)


# @hydra.main(config_path="../configs", config_name="extract_odin.yaml")
@hydra.main(config_path="../configs", config_name="extract_baseline")
def run_experiment(cfg: DictConfig):
    logger.info(cfg)
    out_dir = os.getcwd()
    os.chdir(hydra.utils.get_original_cwd())
    logger.info('torch.cuda.is_available={}'.format(torch.cuda.is_available()))

    # Load pretrained model
    logger.info('Get Model: {} {}'.format(cfg.model, cfg.trainset))
    model = get_model(cfg.model, cfg.trainset, cfg.is_gram if hasattr(cfg, 'is_gram') else False)
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

    # Extract features
    if hasattr(cfg, 'odin'):
        extract_odin_features(model, loaders_dict, cfg.model, cfg.trainset, out_dir, cfg.odin, is_dev_run=cfg.dev_run)
    elif hasattr(cfg, 'is_gram') and cfg.is_gram is True:
        extract_gram_features(model, loaders_dict, out_dir, is_dev_run=cfg.dev_run)
    else:
        extract_baseline_features(model, loaders_dict, out_dir, is_dev_run=cfg.dev_run)
    logger.info('Finished!')


"""
Example of running:
CUDA_VISIBLE_DEVICES=0 python main.py -model densenet -trainset cifar10 -num_workers 0
"""

if __name__ == "__main__":
    run_experiment()
