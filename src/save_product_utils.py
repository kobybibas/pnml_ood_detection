import logging
import os.path as osp

import numpy as np

logger = logging.getLogger(__name__)


def save_last_layer_weights(model, out_dir: str):
    fc = model.fc if hasattr(model, 'fc') else model.linear

    # Save last layer weights
    weights = fc.weight.detach().cpu().numpy()
    bias = fc.bias.detach().cpu().numpy()

    w = np.hstack((np.expand_dims(bias, -1), weights))
    file_name = osp.join(out_dir, 'fc.npy')
    logger.info('Saving to {}'.format(file_name))
    np.save(file_name, w)


def save_train_labels(trainset, trainset_name: str, out_dir: str):
    targets = trainset.targets if hasattr(trainset, 'targets') else trainset.labels
    file_name = osp.join(out_dir, '{}_train_labels.npy'.format(trainset_name))
    logger.info('Saving to {}'.format(file_name))
    np.save(file_name, targets)


def save_products(features_dataset, out_dir: str, set_name: str):
    """
    Save dataset features and outputs logits.
    :param features_dataset:
    :param out_dir:
    :param set_name:
    :return:
    """
    prefix = set_name

    # Save features
    file_name = osp.join(out_dir, '{}_features.npy'.format(prefix))
    logger.info('features shape {}. Saving to {}'.format(features_dataset.data.shape, file_name))
    np.save(file_name, features_dataset.data)

    # Save outputs
    file_name = osp.join(out_dir, '{}_outputs.npy'.format(prefix))
    logger.info('outputs shape {}. Saving to {}'.format(features_dataset.outputs.shape, file_name))
    np.save(file_name, features_dataset.outputs)

    # Save probs
    file_name = osp.join(out_dir, '{}_probs.npy'.format(prefix))
    logger.info('probs shape {}. Saving to {}'.format(features_dataset.probs.shape, file_name))
    np.save(file_name, features_dataset.probs)

    # Save probs
    file_name = osp.join(out_dir, '{}_targets.npy'.format(prefix))
    logger.info('targets shape {}. Saving to {}'.format(features_dataset.targets.shape, file_name))
    np.save(file_name, features_dataset.targets)
