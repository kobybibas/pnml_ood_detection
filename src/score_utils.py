import os.path as osp
import random

import numpy as np
import numpy.linalg as npl
from ood_metrics import calc_metrics
from sklearn.metrics import roc_curve


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    x_max = np.max(x, axis=1, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / e_x.sum(axis=1, keepdims=True)


def add_bias_term(dataset):
    ones = np.ones((len(dataset), 1))
    dataset_with_bias = np.hstack((ones, dataset))
    return dataset_with_bias


def load_features(path: str) -> np.ndarray:
    dataset = np.load(path)
    dataset = add_bias_term(dataset)
    return dataset


def calc_projection_matrices(dataset: np.ndarray) -> np.ndarray:
    # Calc x_Bot
    n, m = dataset.shape
    p = dataset.T @ dataset

    p_parallel = npl.pinv(p)

    p_bot = np.eye(m) - p_parallel @ p
    return p_parallel, p_bot


def calc_regret_on_set(testset, probs, p_parallel, p_bot) -> np.ndarray:
    """
    Calculate the genie probability
    :param testset: tte dataset to evaluate: (n,m)
    :param probs: The model probability of the dataset: (n,)
    :param p_parallel: projection matrix, the parrallel component
    :param p_bot: projection matrix, the orthogonal component
    :return:
    """
    n, n_classes = probs.shape

    # Calc energy of each component
    x_parallel_square = np.array([x @ p_parallel @ x.T for x in testset])
    x_bot_square = np.array([x @ p_bot @ x.T for x in testset])

    #
    x_t_g = np.maximum(x_bot_square, x_parallel_square / (1 + x_parallel_square))
    x_t_g = np.expand_dims(x_t_g, -1)
    x_t_g_repeated = np.repeat(x_t_g, n_classes, axis=1)

    # Genie prediction
    genie_predictions = probs / (probs + (1 - probs) * (probs ** x_t_g_repeated))

    # Regret
    nfs = genie_predictions.sum(axis=1)
    regrets = np.log(nfs) / np.log(n_classes)

    # pNML probability assignment
    pnml_prediction = genie_predictions / np.repeat(
        np.expand_dims(nfs, -1), n_classes, axis=1
    )
    return regrets, pnml_prediction


def load_fc_layer(root: str):
    w = np.load(osp.join(root, "fc.npy"))
    return w


def calc_probs(dataset, w):
    probs = softmax(dataset @ w.T)
    return probs


def transform_features(features):
    # Normalize. The first column is the bias so ignore it
    norm = npl.norm(features[:, 1:], axis=1, keepdims=True)
    features[:, 1:] = features[:, 1:] / norm
    return features


def calc_list_of_dict_mean(dict_list):
    mean_dict = {}
    for key in dict_list[0].keys():
        if isinstance(key, str):
            continue
        mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)

    for key in dict_list[0].keys():
        if isinstance(key, str):
            mean_dict[key] = dict_list[0][key]
    return mean_dict


def split_val_test_idxs(num_samples: int, seed=1):
    random.seed(seed)
    validation_indices = random.sample(range(num_samples), int(0.1 * num_samples))
    test_indices = sorted(list(set(range(num_samples)) - set(validation_indices)))
    return test_indices, validation_indices


def compute_list_of_dict_mean(dict_list: list) -> dict:
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict


def calc_metrics_transformed(ind_score: np.ndarray, ood_score: np.ndarray) -> dict:
    labels = [1] * len(ind_score) + [0] * len(ood_score)
    scores = np.hstack([ind_score, ood_score])

    metric_dict = calc_metrics(scores, labels)
    fpr, tpr, _ = roc_curve(labels, scores)

    metric_dict_transformed = {
        "AUROC": 100 * metric_dict["auroc"],
        "TNR at TPR 95%": 100 * (1 - metric_dict["fpr_at_95_tpr"]),
        "Detection Acc.": 100 * 0.5 * (tpr + 1 - fpr).max(),
    }
    return metric_dict_transformed
