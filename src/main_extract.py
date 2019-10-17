import argparse
import json
import os
import os.path as osp

from loguru import logger

from dataset_utilities import get_dataloaders
from model_utilities import get_model, test_pretrained_model
from pnml_utilities import extract_features, save_products, save_train_labels

"""
Example of running:
CUDA_VISIBLE_DEVICES=0 python main.py -model densenet -trainset cifar10 -num_workers 0
"""


def run_experiment(params):
    os.makedirs(params.data_dir, exist_ok=True)
    os.makedirs(params.output_dir, exist_ok=True)
    os.makedirs(params.logits_dir, exist_ok=True)
    os.makedirs(params.features_dir, exist_ok=True)

    ################
    # Create logger and save params to output folder
    logger.add(osp.join(params.output_dir, 'extract_{}_{}.log'.format(params.model, params.trainset)))
    logger.info(json.dumps(vars(params), indent=4, sort_keys=True))

    ################
    # Load datasets
    logger.info('Load datasets: {}'.format(params.data_dir))
    dataloaders_dict = get_dataloaders(params.trainset, params.data_dir, params.batch_size, params.num_workers)
    logger.info('OOD datasets: {}'.format(dataloaders_dict['ood'].keys()))

    ################
    # Load pretrained model
    logger.info('Get Model: {} {}'.format(params.model, params.trainset))
    model = get_model(params.model_dir, params.model, params.trainset)
    logger.info('Testing pretrained model')
    test_pretrained_model(model, dataloaders_dict['trainset'], dataloaders_dict['testset'])

    for ood_name, dataloader in dataloaders_dict['ood'].items():
        logger.info('Feature extraction for {}'.format(ood_name))
        features_dataset = extract_features(model, dataloader)
        save_products(features_dataset,
                      params.features_dir, params.logits_dir,
                      params.model, params.trainset, ood_name, 'testset')

    # Save trainset
    logger.info('Feature extraction for {}'.format(params.trainset + '_' + 'trainset'))
    features_dataset = extract_features(model, dataloaders_dict['trainset'])
    save_products(features_dataset,
                  params.features_dir, params.logits_dir,
                  params.model, params.trainset, params.trainset, 'trainset')

    # Save train labels
    logger.info('Save train labels for {}'.format(params.trainset))
    save_train_labels(dataloaders_dict['trainset'].dataset, params.trainset, params.features_dir)
    save_train_labels(dataloaders_dict['trainset'].dataset, params.trainset, params.logits_dir)

    logger.info('Finished!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Applications of Deep PNML')
    parser.add_argument('-model',
                        help='Testset to evaluate',
                        default='densenet',
                        choices=['densenet', 'resnet'],
                        type=str)
    parser.add_argument('-trainset',
                        help='Trainset of the model',
                        default='cifar10',
                        choices=['cifar10', 'cifar100'],
                        type=str)
    parser.add_argument('-output_dir',
                        help='Trainset of the model',
                        default=osp.join('..', 'output'),
                        type=str)
    parser.add_argument('-data_dir',
                        help='Trainset of the model',
                        default=osp.join('..', 'data'),
                        type=str)
    parser.add_argument('-model_dir',
                        help='Trainset of the model',
                        default=osp.join('..', 'models'),
                        type=str)
    parser.add_argument('-logits_dir',
                        help='Trainset of the model',
                        default=osp.join('..', 'output', 'logits'),
                        type=str)
    parser.add_argument('-features_dir',
                        help='Trainset of the model',
                        default=osp.join('..', 'output', 'features'),
                        type=str)
    parser.add_argument('-batch_size',
                        default=128,
                        help='Number of CPU workers',
                        type=int)
    parser.add_argument('-num_workers',
                        default=4,
                        help='Number of CPU workers',
                        type=int)

    args = parser.parse_args()
    run_experiment(args)
