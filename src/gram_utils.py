import logging
import os.path as osp

import numpy as np
import torch
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

from dataset_utils import FeaturesDataset
from model_arch_utils.densenet_gram import DenseNet3Gram
from model_arch_utils.resnet_gram import ResNetGram
from save_product_utils import save_products

logger = logging.getLogger(__name__)


def extract_gram_features(model, loaders_dict: dict, out_dir: str, is_dev_run: bool = False):
    data_name = 'trainset'
    train_loader = loaders_dict.pop(data_name)
    logger.info('Feature extraction for {}'.format(data_name))
    features_dataset, gram_trainset_features = gram_extract_features_from_loader(model, train_loader,
                                                                                 is_dev_run=is_dev_run)
    save_products(features_dataset, out_dir, data_name)

    # Compute per class Gram features min and max of the training set
    gram_mins, gram_maxs = compute_minmaxs_train(gram_trainset_features, features_dataset.probs)
    gram_minmaxs_path = osp.join(out_dir, 'trainset_gram_mins_and_maxs.npy')
    np.save(gram_minmaxs_path, {'gram_mins': gram_mins, 'gram_maxs': gram_maxs})

    # Extract features for all dataset: trainset, ind_testset and ood_testsets
    for data_name, loader in loaders_dict.items():
        logger.info('Feature extraction for {}'.format(data_name))
        features_dataset, gram_features = gram_extract_features_from_loader(model, loader, is_dev_run=is_dev_run)
        save_products(features_dataset, out_dir, data_name)

        # Compute deviations of the test gram features from the min and max of the trainset
        deviations = compute_deviations(gram_features, features_dataset.probs, gram_mins, gram_maxs)
        gram_deviations_path = osp.join(out_dir, f"{data_name}_gram_deviations.npy")
        np.save(gram_deviations_path, deviations)
        logger.info('')


def gram_extract_features_from_loader(model: torch.nn.Module, dataloader: data.DataLoader,
                                      is_dev_run: bool = False) -> (FeaturesDataset, torch.Tensor):
    features_list, labels_list, outputs_list, prob_list = [], [], [], []
    gram_features_list = None
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()

    assert isinstance(model, ResNetGram) or isinstance(model, DenseNet3Gram)
    model.set_collecting(True)
    with torch.no_grad():
        for batch_n, (images, labels) in enumerate(tqdm(dataloader)):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=-1)

            # Get Features
            features_list.append(model.get_features().cpu().detach())
            labels_list.append(labels.cpu())
            outputs_list.append(outputs.cpu().detach())
            prob_list.append(probs.cpu().detach())

            gram_features_batch = model.collect_gram_features()
            if gram_features_list is None:
                gram_features_list = gram_features_batch
            else:
                for l in range(len(gram_features_list)):
                    gram_features_list[l] = np.vstack((gram_features_list[l], gram_features_batch[l]))

            if is_dev_run is True and batch_n > 1:
                break

    features_dataset = FeaturesDataset(features_list, labels_list, outputs_list, prob_list,
                                       transform=transforms.Compose([transforms.Lambda(lambda x: x)]))

    torch.cuda.empty_cache()
    return features_dataset, gram_features_list


def relu(x):
    return x * (x > 0)


def get_deviations(features_per_layer_list, mins, maxs) -> np.ndarray:
    deviations = []
    for layer_num, features in enumerate(features_per_layer_list):
        layer_t = features  # features.transpose(2, 0, 1)  # [sample,power,value].
        mins_expand = np.expand_dims(mins[layer_num], axis=0)
        maxs_expand = np.expand_dims(maxs[layer_num], axis=0)

        # Divide each sample by the same min of the layer-power-feature
        devs_l = (relu(mins_expand - layer_t) / np.abs(mins_expand + 10 ** -6)).sum(axis=(1, 2))
        devs_l += (relu(layer_t - maxs_expand) / np.abs(maxs_expand + 10 ** -6)).sum(axis=(1, 2))
        deviations.append(np.expand_dims(devs_l, 1))
    deviations = np.concatenate(deviations, axis=1)  # shape=[num_samples,num_layer]
    return deviations


def compute_minmaxs_train(gram_feature_layers, probs):
    predictions = np.argmax(probs, axis=1)

    predictions_unique = np.unique(predictions)
    predictions_unique.sort()

    # Initialize outputs:
    mins, maxs = {}, {}

    # Iterate on labels
    for c in tqdm(predictions_unique, desc='compute_minmaxs_train'):
        # Extract samples that are predicted as c
        class_idxs = np.where(c == predictions)[0]
        gram_features_c = [gram_feature_in_layer_i[class_idxs] for gram_feature_in_layer_i in gram_feature_layers]

        # compute min and max of the gram features (per layer per power)
        mins_per_class = [np.array(layer).min(axis=0) for layer in gram_features_c]  # shape=[layers,powers,features]
        maxs_per_class = [np.array(layer).max(axis=0) for layer in gram_features_c]

        # Save
        mins[c] = mins_per_class
        maxs[c] = maxs_per_class

    return mins, maxs


def compute_deviations(gram_feature_layers, probs, gram_mins, gram_maxs):
    # Initialize outputs
    deviations = []

    max_probs = probs.max(axis=1)
    predictions = probs.argmax(axis=1)

    # Iterate on labels
    predictions_unique = np.unique(predictions)
    predictions_unique.sort()

    for c in tqdm(predictions_unique, desc='compute_deviations'):
        # Initialize per class
        class_idxs = np.where(c == predictions)[0]
        gram_features_per_class = [gram_feature_layer[class_idxs] for gram_feature_layer in gram_feature_layers]
        max_probs_c = max_probs[predictions == c]

        deviations_c = get_deviations(gram_features_per_class, mins=gram_mins[c], maxs=gram_maxs[c])
        deviations_c /= max_probs_c[:, np.newaxis]

        deviations.append(deviations_c)

    deviations = np.concatenate(deviations, axis=0)

    return deviations
