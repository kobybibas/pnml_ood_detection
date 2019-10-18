import json
import os
import os.path as osp
from argparse import ArgumentParser

import numpy as np
import sys
import torch
from loguru import logger as loguru_logger
from pytorchcv.model_provider import get_model as ptcv_get_model

from logger_utilities import MyLogger
from train_utilities import ModelWrapper
from train_utilities import MyTrainer as Trainer


def run(params):
    os.makedirs(params.data_dir, exist_ok=True)
    os.makedirs(params.model_dir, exist_ok=True)
    os.makedirs(params.output_dir, exist_ok=True)

    logger = MyLogger(loguru_logger)
    logger.add(osp.join(params.output_dir, "train_{}_{}.log".format(params.model, params.dataset)),
               backtrace=True, diagnose=True, rotation="50 MB")
    logger.info(json.dumps(vars(params), sort_keys=True, indent=4))

    # For reproducibility
    # np.random.seed(0)
    # torch.manual_seed(0)

    # Load model
    model = None
    if params.model == 'densenet':
        model = ptcv_get_model("densenet100_k12_bc_%s" % params.dataset, pretrained=True)
    elif params.model == 'resnet':
        model = ptcv_get_model("wrn28_10_%s" % params.dataset, pretrained=True)
    else:
        logger.error('model_name {} is not supported.'.format(params.model))

    # Reset last layer for quick training
    model.output = torch.nn.Linear(model.output.in_features, model.output.out_features)

    wrapper_h = ModelWrapper(model, params)
    trainer = Trainer(logger=logger,
                      show_progress_bar=False,
                      default_save_path=params.model_dir,
                      min_nb_epochs=params.epochs - 1,
                      max_nb_epochs=params.epochs,
                      row_log_interval=50,
                      gpus=None if sys.platform == 'darwin' or not torch.cuda.is_available() else 1,
                      checkpoint_callback=False)

    trainer.fit(wrapper_h)

    save_path = osp.join(params.model_dir, '{}_{}.pth'.format(params.model, params.dataset))
    logger.info('Saving model: {}'.format(save_path))
    torch.save(wrapper_h.model.state_dict(), save_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'],
                        help='Dataset to train on (default: cifar10)')
    parser.add_argument('-model', type=str, default='densenet',
                        choices=['densenet', 'resnet'],
                        help='Model architecture to use (default: densenet)')
    parser.add_argument('-batch_size', type=int, default=128,
                        help='input batch size for training (default: 64)')
    parser.add_argument('-epochs', type=int, default=20,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('-lr', type=float, default=1e-3,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('-step_size', type=float, default=6,
                        help='ADAM lr scheduler step size (default: 30)')
    parser.add_argument('-weight_decay', type=float, default=0.0005,
                        help='l2 regularization (default: ﻿0.0005)')
    parser.add_argument('-num_workers', type=int, default=4,
                        help='Number of CPU workers (default: 4)')
    parser.add_argument('-data_dir', type=str, default=osp.join('..', 'data'),
                        help='Data dir to which the dataset will be downloaded')
    parser.add_argument('-model_dir', type=str, default=osp.join('..', 'models'),
                        help='Mode directory to which the model will be saved')
    parser.add_argument('-output_dir', type=str, default=osp.join('..', 'output'),
                        help='Logs and products.')
    parser.add_argument('-is_svd', type=bool, action='store_true',
                        default=False,
                        help='')

    args = parser.parse_args()

    run(args)
