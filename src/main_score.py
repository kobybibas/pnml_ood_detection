import argparse
import json
import os
import os.path as osp

import numpy as np
from loguru import logger

from dataset_utilities import testsets_name
from score_utilities import load_trainset_features, load_test_products, project_testset, decompose_trainset


def run_experiment(params):
    # ------------ #
    # Preparations
    # ------------ #
    os.makedirs(params.output_dir, exist_ok=True)
    os.makedirs(params.score_dir, exist_ok=True)
    logger.add(osp.join(params.output_dir, 'score_{}_{}.log'.format(params.model, params.trainset)))
    logger.info(json.dumps(vars(params), indent=4, sort_keys=True))

    assert os.path.isdir(params.logits_dir)
    assert os.path.isdir(params.features_dir)

    # ---------------------- #
    # Load trainset features
    # ---------------------- #
    logger.info('Load trainset features: {}'.format(params.features_dir))
    trainset_features, trainset_labels = load_trainset_features(params.model,
                                                                params.trainset,
                                                                params.features_dir)
    logger.info('Trainset shape: {}'.format(trainset_features.shape))

    # ---------------------------- #
    # Decompose trainset per class
    # ---------------------------- #
    logger.info('Decompose trainset')
    svd_list = decompose_trainset(trainset_features, trainset_labels)
    eigenvalues = [eta for _, eta, _ in svd_list]
    save_file = osp.join(params.score_dir, '{}_{}_eigenvalues.npy'.format(params.model, params.trainset))
    np.save(save_file, eigenvalues)
    logger.info('Finish SVD. Saved to: {}'.format(save_file))

    # ------------------------------- #
    # Calculate In Distribution score
    # ------------------------------- #
    testset_features, testset_pred = load_test_products(params.model, params.trainset, params.trainset,
                                                        params.features_dir, params.logits_dir)
    ind_score = project_testset(testset_features, testset_pred, svd_list)
    save_file = osp.join(params.score_dir, '{}_{}_distance_ind.txt'.format(params.model, params.trainset))
    np.savetxt(save_file, ind_score, delimiter='\n')
    logger.info('Finish Calc In-Distribution score. Saved to: {}'.format(save_file))

    # -------------------------------------- #
    # Iterate on Out of Distribution testsets
    # -------------------------------------- #
    testsets_name_iter = ['cifar10'] + testsets_name if params.trainset == 'cifar100' else ['cifar100'] + testsets_name
    for ood_num, ood_name in enumerate(testsets_name_iter):
        logger.info('[{}/{}] Compute score for {}'.format(ood_num, len(testsets_name_iter) - 1, ood_name))

        testset_features, testset_pred = load_test_products(params.model, params.trainset, ood_name,
                                                            params.features_dir, params.logits_dir)

        ood_score = project_testset(testset_features, testset_pred, svd_list)

        # ----------- #
        # Save score
        # ----------- #
        save_file = osp.join(params.score_dir,
                             '{}_{}_distance_ood_{}.txt'.format(params.model, params.trainset, ood_name))
        np.savetxt(save_file, ood_score, delimiter='\n')
        logger.info('Finish Calc OOD score. Saved to: {}'.format(save_file))
        logger.info('Finished!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Applications of Deep PNML')
    parser.add_argument('-model',
                        help='Model architecture name',
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
    parser.add_argument('-score_dir',
                        help='Trainset of the model',
                        default=osp.join('..', 'output', 'score'),
                        type=str)
    parser.add_argument('-logits_dir',
                        help='Trainset of the model',
                        default=osp.join('..', 'output', 'logits'),
                        type=str)
    parser.add_argument('-features_dir',
                        help='Trainset of the model',
                        default=osp.join('..', 'output', 'features'),
                        type=str)

    args = parser.parse_args()
    run_experiment(args)
