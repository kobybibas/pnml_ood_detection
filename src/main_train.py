import os
import os.path as osp
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, RunningAverage
from loguru import logger
from pytorchcv.model_provider import get_model as ptcv_get_model
from torch.optim import Adam
from torch.optim import lr_scheduler

from dataset_utilities import create_cifar10_dataloaders, create_cifar100_dataloaders


def run(params):
    os.makedirs(params.data_dir, exist_ok=True)
    os.makedirs(params.model_dir, exist_ok=True)
    os.makedirs(params.output_dir, exist_ok=True)
    logger.add(osp.join(params.output_dir, "{}_{}_train.log".format(params.model_name, params.dataset_name)),
               backtrace=True, diagnose=True, rotation="50 MB")
    logger.info(params)

    # Load model
    model = None
    if params.model_name == 'densenet':
        model = ptcv_get_model("densenet100_k12_bc_%s" % params.dataset_name, pretrained=True)
    elif params.model_name == 'resnet':
        model = ptcv_get_model("wrn28_10_%s" % params.dataset_name, pretrained=True)
    else:
        ValueError('model_name {} is not supported.'.format(params.model_name))

    # Reset last layer
    model.output = torch.nn.Linear(model.output.in_features, model.output.out_features)

    # Load dataset
    if params.dataset_name == 'cifar10':
        train_loader, val_loader, _ = create_cifar10_dataloaders(params.data_dir,
                                                                 params.batch_size,
                                                                 params.num_workers,
                                                                 True)
    elif params.dataset_name == 'cifar100':
        train_loader, val_loader, _ = create_cifar100_dataloaders(params.data_dir,
                                                                  params.batch_size,
                                                                  params.num_workers,
                                                                  True)
    else:
        ValueError('dataset_name {} is not supported.'.format(params.dataset_name))
    assert isinstance(model, torch.nn.Module)
    assert isinstance(train_loader, torch.utils.data.DataLoader)
    assert isinstance(val_loader, torch.utils.data.DataLoader)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    optimizer = Adam(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, params.step_size)

    trainer = create_supervised_trainer(model, optimizer, F.cross_entropy, device=device)
    evaluator = create_supervised_evaluator(model, metrics={'accuracy': Accuracy(),
                                                            'nll': Loss(F.cross_entropy)},
                                            device=device)

    RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')
    pbar = ProgressBar(persist=True)
    pbar.attach(trainer, metric_names='all')
    avg_val_acc = [0]

    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        pbar.log_message(
            "Training Results - Epoch: {}  Avg accuracy: {:.3f} Avg loss: {:.3f}"
                .format(engine.state.epoch, avg_accuracy, avg_nll)
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_val_acc[0] = 100 * avg_accuracy
        lr = get_lr(optimizer)
        avg_nll = metrics['nll']
        msg = "Validation Results - Epoch: {}  Avg accuracy: {:.3f} Avg loss: {:.3f}, lr={:.6f}".format(
            engine.state.epoch,
            avg_accuracy, avg_nll,
            lr)
        pbar.log_message(msg)
        logger.info(msg)
        pbar.n = pbar.last_print_n = 0
        scheduler.step()

    trainer.run(train_loader, max_epochs=params.epochs)
    save_path = osp.join(params.model_dir, '{}_{}_acc_{:.2f}.pth'.format(params.model_name,
                                                                         params.dataset_name,
                                                                         avg_val_acc[0]))
    logger.info('Saving model: {}'.format(save_path))
    torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-dataset_name', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'],
                        help='Dataset to train on (default: cifar10)')
    parser.add_argument('-model_name', type=str, default='densenet',
                        choices=['densenet', 'resnet'],
                        help='Model architecture to use (default: densenet)')
    parser.add_argument('-batch_size', type=int, default=128,
                        help='input batch size for training (default: 64)')
    parser.add_argument('-epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('-lr', type=float, default=1e-3,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('-step_size', type=float, default=3,
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

    args = parser.parse_args()

    run(args)
