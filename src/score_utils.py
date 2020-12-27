import os
import os.path as osp
import time
from glob import glob

import numpy as np
import numpy.linalg as npl
import pandas as pd
import tqdm
from ood_metrics import calc_metrics
from tqdm import tqdm

ood_names_dict = {
    "TinyImageNet (crop)": "Imagenet",
    "TinyImageNet (resize)": "Imagenet_resize",
    "LSUN (crop)": "LSUN",
    "LSUN (resize)": "LSUN_resize",
    "iSUN": "iSUN",
    "Uniform": "Uniform",
    "Gaussian": "Gaussian",
}


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

    rank = min(n, m)
    p_bot = np.eye(rank) - p_parallel @ p
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
    pnml_prediction = genie_predictions / np.repeat(np.expand_dims(nfs, -1), n_classes, axis=1)
    return regrets, pnml_prediction


def load_ood_products(root: str, trainset_name: str) -> dict:
    feature_paths = glob(osp.join(root, f"*_features.npy"))
    feature_paths.sort()

    prob_paths = glob(osp.join(root, f"*_probs.npy"))
    prob_paths.sort()

    ood_dict = {}
    for prob_path, feature_path in zip(prob_paths, feature_paths):

        # Skip IND
        if trainset_name + '_features.npy' in feature_path or 'trainset' in feature_path:
            continue

        # Get OOD name
        ood_name = osp.basename(feature_path)
        ood_name = ood_name.replace("_features.npy", '')

        ood_dict[ood_name] = {'feature_path': feature_path,
                              'prob_path': prob_path}
    return ood_dict


def load_ind_products(root: str, trainset_name: str):
    ind_features = load_features(osp.join(root, f"{trainset_name}_features.npy"))
    testset_ind_labels = np.load(osp.join(root, f"{trainset_name}_targets.npy"))
    ind_probs = np.load(osp.join(root, f"{trainset_name}_probs.npy"))
    return ind_features, testset_ind_labels, ind_probs


def load_fc_layer(root: str):
    w = np.load(osp.join(root, 'fc.npy'))
    return w


def calc_probs(dataset, w):
    probs = softmax(dataset @ w.T)
    return probs


def transform_features(features):
    # Normalize. The first column is the bias so ignore it
    norm = npl.norm(features[:, 1:], axis=1, keepdims=True)
    features[:, 1:] = features[:, 1:] / norm
    return features


def rename_df(df, method_name, model_name, trainset_name):
    df_new = df.set_index(["ood_dataset"], inplace=False).reindex(["iSUN",
                                                                   "LSUN_resize",
                                                                   "Imagenet_resize",
                                                                   "svhn",
                                                                   "Imagenet",
                                                                   "LSUN",
                                                                   "Uniform",
                                                                   "Gaussian"]).reset_index()
    df_new['model_name'] = f'{model_name}_{trainset_name}'
    df_new = df_new.set_index(["model_name", "ood_dataset"], inplace=False)

    df_new = df_new.round(1)
    df_new = df_new[['auroc']]
    df_new = df_new.rename(columns={"auroc": method_name})
    return df_new


def calc_pnml_pref(ood_name, regrets_ind, regrets_ood):
    labels = [1] * len(regrets_ind) + [0] * len(regrets_ood)
    scores = 1 - np.append(regrets_ind, regrets_ood)
    performance_dict = calc_metrics(scores, labels)
    performance_dict = {key: 100 * value for key, value in performance_dict.items()}
    performance_dict["ood_dataset"] = ood_name
    return performance_dict


def calc_max_prob_pref(ood_name: str, probs_ind_path: str, probs_ood_path: str):
    probs_ind = np.load(probs_ind_path)
    probs_ood = np.load(probs_ood_path)

    max_probs_ind = np.max(probs_ind, axis=1)
    max_probs_ood = np.max(probs_ood, axis=1)

    # Calc metric max prob
    labels = [1] * len(max_probs_ind) + [0] * len(max_probs_ood)
    scores = np.append(max_probs_ind, max_probs_ood)
    performance_dict = calc_metrics(scores, labels)
    performance_dict = {key: 100 * value for key, value in performance_dict.items()}
    performance_dict["ood_dataset"] = ood_name
    return performance_dict


def calc_performance_odin(root: str, trainset_name: str) -> (pd.DataFrame, pd.DataFrame):
    # Load trainset products
    trainset_features = load_features(osp.join(root, f"trainset_features.npy"))
    trainset_features = transform_features(trainset_features)
    p_parallel, p_bot = calc_projection_matrices(trainset_features)
    w = load_fc_layer(root)

    ood_roots = glob(osp.join(root, '*'))
    ood_roots = filter(os.path.isdir, ood_roots)

    pnml_performance_list, max_prob_performance_list = [], []
    for ood_root in tqdm(ood_roots):
        # Load IND products
        ind_features, ind_labels, ind_probs = load_ind_products(ood_root, trainset_name)
        ind_prob_path = osp.join(ood_root, f"{trainset_name}_probs.npy")

        # Load OOD products
        ood_dict = load_ood_products(ood_root, trainset_name)
        ood_name, ood_dict = next(iter(ood_dict.items()))

        # Max prob
        performance_dict = calc_max_prob_pref(ood_name, ind_prob_path, ood_dict['prob_path'])
        max_prob_performance_list.append(performance_dict)

        # Compute pNML

        # ind
        ind_features = transform_features(ind_features)
        ind_probs = calc_probs(ind_features, w)
        regrets_ind, pnml_prediction = calc_regret_on_set(ind_features, ind_probs, p_parallel, p_bot)

        # ood
        ood_features = load_features(ood_dict['feature_path'])
        ood_features = transform_features(ood_features)
        ood_probs = calc_probs(ood_features, w)
        regrets_ood, _ = calc_regret_on_set(ood_features, ood_probs, p_parallel, p_bot)
        performance_dict = calc_pnml_pref(ood_name, regrets_ind, regrets_ood)
        pnml_performance_list.append(performance_dict)

    return pd.DataFrame(pnml_performance_list), pd.DataFrame(max_prob_performance_list)


def calc_performance_baseline(root: str, trainset_name: str) -> (pd.DataFrame, pd.DataFrame):
    # Initialize output
    pnml_performance_list, max_prob_performance_list = [], []

    # Load trainset products
    trainset_features = load_features(osp.join(root, f"trainset_features.npy"))
    trainset_features = transform_features(trainset_features)
    p_parallel, p_bot = calc_projection_matrices(trainset_features)
    w = load_fc_layer(root)

    # Load IND products
    ind_features, ind_labels, ind_probs = load_ind_products(root, trainset_name)
    ind_prob_path = osp.join(root, f"{trainset_name}_probs.npy")

    # Score pNML
    ind_features = transform_features(ind_features)
    ind_probs = calc_probs(ind_features, w)
    regrets_ind, pnml_prediction = calc_regret_on_set(ind_features, ind_probs, p_parallel, p_bot)

    # Load OOD products
    ood_dict = load_ood_products(root, trainset_name)

    for ood_name, ood_dict in tqdm(ood_dict.items()):
        # Calc metric max prob
        performance_dict = calc_max_prob_pref(ood_name, ind_prob_path, ood_dict['prob_path'])
        max_prob_performance_list.append(performance_dict)

        # Compute pNML
        ood_features = load_features(ood_dict['feature_path'])
        ood_features = transform_features(ood_features)
        ood_probs = calc_probs(ood_features, w)

        # Calc metric pnml
        regrets_ood, _ = calc_regret_on_set(ood_features, ood_probs, p_parallel, p_bot)
        performance_dict = calc_pnml_pref(ood_name, regrets_ind, regrets_ood)
        pnml_performance_list.append(performance_dict)

    return pd.DataFrame(pnml_performance_list), pd.DataFrame(max_prob_performance_list)


if __name__ == '__main__':
    desired_width = 320
    pd.set_option('display.width', desired_width)
    pd.set_option('display.max_columns', 10)

    roots = [
        [osp.join('..', 'outputs', 'baseline_resnet_cifar10_20201225_082002'),
         osp.join('..', 'outputs', 'odin_resnet_cifar10_20201227_103641')
         ],
        [osp.join('..', 'outputs', 'baseline_resnet_cifar100_20201225_082335'),
         osp.join('..', 'outputs', 'odin_resnet_cifar100_20201227_105829')
         ],
        [osp.join('..', 'outputs', 'baseline_densenet_cifar10_20201225_080842'),
         osp.join('..', 'outputs', 'odin_densenet_cifar10_20201227_095311')
         ],
        [osp.join('..', 'outputs', 'baseline_densenet_cifar100_20201225_081421'),
         osp.join('..', 'outputs', 'odin_densenet_cifar100_20201227_101407')
         ]
    ]

    t0 = time.time()
    model_results = {}
    for i, roots_i in enumerate(roots):
        t1 = time.time()
        dfs = []
        for root_i in roots_i:
            method_i, model_name_i, trainset_name_i = osp.basename(root_i).split('_')[0:3]
            if method_i == 'baseline':
                pnml_df, max_prob_df = calc_performance_baseline(root_i, trainset_name_i)
            elif method_i == 'odin':
                pnml_df, max_prob_df = calc_performance_odin(root_i, trainset_name_i)
            else:
                raise ValueError(f'{method_i} is not supported')

            pnml_df = rename_df(pnml_df, method_i + '+pNML', model_name_i, trainset_name_i)
            max_prob_df = rename_df(max_prob_df, method_i, model_name_i, trainset_name_i)

            df = pd.concat((max_prob_df, pnml_df), axis=1)
            dfs.append(df)
        key_name = '{}_{}'.format(model_name_i, trainset_name_i)
        df = pd.concat(dfs, axis=1)
        model_results[key_name] = df

        print('[{}/{}] {} {} in {:.2f} s'.format(i, len(roots) - 1, model_name_i, trainset_name_i, time.time() - t1))
        print(model_results[key_name])
        print()

    print('Finish in {:.2f} sec'.format(time.time() - t0))
    result_df = pd.concat(list(model_results.values()), axis=0)
    print(result_df)
    result_df.to_csv('../outputs/results.csv')
