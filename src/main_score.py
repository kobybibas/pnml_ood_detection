import os
import os.path as osp
import time
from glob import glob

import numpy as np
import pandas as pd
import tqdm
from tqdm import tqdm

from score_utils import calc_list_of_dict_mean
from score_utils import calc_metrics_transformed as calc_metrics
from score_utils import calc_regret_on_set, load_fc_layer, calc_probs, split_val_test_idxs
from score_utils import load_features, transform_features, calc_projection_matrices

ood_names_dict = {
    "TinyImageNet (crop)": "Imagenet",
    "TinyImageNet (resize)": "Imagenet_resize",
    "LSUN (crop)": "LSUN",
    "LSUN (resize)": "LSUN_resize",
    "iSUN": "iSUN",
    "Uniform": "Uniform",
    "Gaussian": "Gaussian",
}


def load_ood_products(root: str, trainset_name: str) -> dict:
    feature_paths = glob(osp.join(root, f"*_features.npy"))
    feature_paths.sort()

    prob_paths = glob(osp.join(root, f"*_probs.npy"))
    prob_paths.sort()

    ood_dict = {}
    for prob_path, feature_path in zip(prob_paths, feature_paths):

        # Skip IND
        if 'trainset' in feature_path:
            continue
        if trainset_name + '_features.npy' in feature_path:
            continue

        # Get OOD name
        ood_name = osp.basename(feature_path)
        ood_name = ood_name.replace("_features.npy", '')

        ood_dict[ood_name] = {'feature_path': feature_path,
                              'prob_path': prob_path,
                              'gram_path': osp.join(root, f"{ood_name}_gram.npy"),
                              'gram_dev_path': osp.join(root, f"{ood_name}_gram_deviations.npy")}
    return ood_dict


def load_ind_products(root: str, trainset_name: str):
    ind_features = load_features(osp.join(root, f"{trainset_name}_features.npy"))
    testset_ind_labels = np.load(osp.join(root, f"{trainset_name}_targets.npy"))
    ind_probs = np.load(osp.join(root, f"{trainset_name}_probs.npy"))
    return ind_features, testset_ind_labels, ind_probs


def reorder_df_index(df):
    df_new = df.reindex(["iSUN",
                         "LSUN_resize",
                         "Imagenet_resize",
                         "svhn",
                         "Imagenet",
                         "LSUN",
                         "Uniform",
                         "Gaussian",
                         "cifar10",
                         "cifar100"]).reset_index()
    return df_new


def create_nested_df(method_dicts: dict) -> pd.DataFrame:
    # Change nested dict: metric -> method -> values
    metric_method_dict = {}
    for method, metrics_dict in method_dicts.items():
        for metric_name, metric_values in metrics_dict.items():
            if not metric_name in metric_method_dict:
                metric_method_dict[metric_name] = {}
            metric_method_dict[metric_name][method] = metric_values

    # Create nested df
    dict_of_df = {k: pd.DataFrame(v) for k, v in metric_method_dict.items()}
    ood_names_df = dict_of_df.pop('ood_dataset')
    for metric_name, df_j in dict_of_df.items():
        method = df_j.columns[0]
        ood_names = ood_names_df[method]
        df_j.index = ood_names
        df_j.index.name = 'ood_dataset'
    df_concat = pd.concat(dict_of_df, axis=1)
    df_concat['model_name'] = f'{model_name_i}_{trainset_name_i}'
    df_concat = reorder_df_index(df_concat)
    return df_concat


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


def calc_gram_pref(ood_name: str, deviation_ind, deviation_ood):
    labels = [1] * len(deviation_ind) + [0] * len(deviation_ood)
    scores = -np.append(deviation_ind, deviation_ood)
    performance_dict = calc_metrics(scores, labels)
    performance_dict = {key: 100 * value for key, value in performance_dict.items()}
    performance_dict["ood_dataset"] = ood_name
    return performance_dict


def calc_performance_odin(root: str, trainset_name: str) -> (dict, dict, list):
    # Load trainset products
    trainset_features = load_features(osp.join(root, f"trainset_features.npy"))
    trainset_features = transform_features(trainset_features)
    p_parallel, p_bot = calc_projection_matrices(trainset_features)
    w = load_fc_layer(root)

    ood_roots = glob(osp.join(root, '*'))
    ood_roots = filter(os.path.isdir, ood_roots)

    pnml_performance_list, odin_performance_list = [], []
    for ood_root in tqdm(list(ood_roots), desc='odin: ood'):
        # Load IND products
        ind_features, ind_labels, ind_probs = load_ind_products(ood_root, trainset_name)
        ind_prob_path = osp.join(ood_root, f"{trainset_name}_probs.npy")

        # Load OOD products
        ood_dict = load_ood_products(ood_root, trainset_name)
        ood_name, ood_dict = next(iter(ood_dict.items()))

        # Max prob
        performance_dict = calc_max_prob_pref(ood_name, ind_prob_path, ood_dict['prob_path'])
        odin_performance_list.append(performance_dict)

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

    pnml_dict = pd.DataFrame(pnml_performance_list).to_dict('list')
    odin_dict = pd.DataFrame(odin_performance_list).to_dict('list')
    return pnml_dict, odin_dict


def calc_performance_gram(root: str, trainset_name: str) -> (dict, dict):
    # Initialize output
    pnml_performance_list, gram_performance_list = [], []

    # Load trainset products
    trainset_features = load_features(osp.join(root, f"trainset_features.npy"))
    trainset_features = transform_features(trainset_features)
    p_parallel, p_bot = calc_projection_matrices(trainset_features)
    w = load_fc_layer(root)

    # Load IND products
    ind_features, ind_labels, probs_ind = load_ind_products(root, trainset_name)

    # Score gram
    ind_gram_dev_path = osp.join(root, f"{trainset_name}_gram_deviations.npy")
    gram_deviations_ind = np.load(ind_gram_dev_path, allow_pickle=True)

    # Score pNML
    ind_features = transform_features(ind_features)
    probs_ind = calc_probs(ind_features, w)

    # Load OOD products
    ood_dict = load_ood_products(root, trainset_name)
    for ood_name, ood_dict in tqdm(ood_dict.items(), desc='gram: ood'):

        # Load ood products
        ood_features = load_features(ood_dict['feature_path'])
        ood_features = transform_features(ood_features)
        ood_probs = calc_probs(ood_features, w)
        deviations_ood = np.load(ood_dict['gram_dev_path'])

        gram_perf_dict_list, pnml_df_list = [], []
        for seed in range(1, 11):
            # Split
            test_indices, val_indices = split_val_test_idxs(len(gram_deviations_ind), seed)
            deviations_ind_val = gram_deviations_ind[val_indices]
            deviations_ind_test = gram_deviations_ind[test_indices]

            # Compute Gram
            dev_norm_sum = deviations_ind_val.sum(axis=0, keepdims=True) + 10 ** -7
            ind_deviations_norm = (deviations_ind_test / dev_norm_sum).mean(axis=1)
            ood_deviations_norm = (deviations_ood / dev_norm_sum).mean(axis=1)

            # Calc metric pnml
            performance_dict = calc_gram_pref(ood_name, ind_deviations_norm, ood_deviations_norm)
            gram_perf_dict_list.append(performance_dict)

            # Compute pNML
            dev_norm_std = deviations_ind_val.std(axis=0, keepdims=True)
            dev_norm_std[dev_norm_std == 0.0] = 1.0
            ind_deviations_norm = np.sqrt((deviations_ind_test / dev_norm_std).mean(axis=1, keepdims=True))
            ood_deviations_norm = np.sqrt((deviations_ood / dev_norm_std).mean(axis=1, keepdims=True))
            regrets_ind, pnml_prediction = calc_regret_on_set(ind_features[test_indices] * ind_deviations_norm,
                                                              probs_ind[test_indices],
                                                              p_parallel, p_bot)

            # Calc metric pnml
            regrets_ood, _ = calc_regret_on_set(ood_features * ood_deviations_norm,
                                                ood_probs, p_parallel, p_bot)

            performance_dict = calc_pnml_pref(ood_name, regrets_ind, regrets_ood)
            pnml_df_list.append(performance_dict)

        # Store average performance
        gram_performance_list.append(calc_list_of_dict_mean(gram_perf_dict_list))
        pnml_performance_list.append(calc_list_of_dict_mean(pnml_df_list))

    pnml_dict = pd.DataFrame(pnml_performance_list).to_dict('list')
    gram_dict = pd.DataFrame(gram_performance_list).to_dict('list')
    return pnml_dict, gram_dict


def calc_performance_baseline(root: str, trainset_name: str) -> (dict, dict):
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

    for ood_name, ood_dict in tqdm(ood_dict.items(), desc='baseline: ood'):
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

    pnml_dict = pd.DataFrame(pnml_performance_list).to_dict('list')
    max_prob_dict = pd.DataFrame(max_prob_performance_list).to_dict('list')
    return pnml_dict, max_prob_dict


if __name__ == '__main__':
    desired_width = 320
    pd.set_option('display.width', desired_width)
    pd.set_option('display.max_columns', 10)

    roots = [
        # Resnet CIFAR10
        [osp.join('..', 'outputs', 'baseline_resnet_cifar10_20201225_082002'),
         osp.join('..', 'outputs', 'odin_resnet_cifar10_20201227_103641'),
         osp.join('..', 'outputs', 'gram_resnet_cifar10_20210101_194316'),
         ],
        # Resnet CIFAR100
        [osp.join('..', 'outputs', 'baseline_resnet_cifar100_20201225_082335'),
         osp.join('..', 'outputs', 'odin_resnet_cifar100_20201227_105829'),
         osp.join('..', 'outputs', 'gram_resnet_cifar100_20210101_194313'),
         ],
        # Densenet CIFAR10
        [osp.join('..', 'outputs', 'baseline_densenet_cifar10_20201225_080842'),
         osp.join('..', 'outputs', 'odin_densenet_cifar10_20201227_095311'),
         osp.join('..', 'outputs', 'gram_densenet_cifar10_20210102_101211'),
         ],
        # Densenet CIFAR100
        [osp.join('..', 'outputs', 'baseline_densenet_cifar100_20201225_081421'),
         osp.join('..', 'outputs', 'odin_densenet_cifar100_20201227_101407'),
         osp.join('..', 'outputs', 'gram_densenet_cifar100_20210102_101209')
         ],
        # Densenet SVHN
        [osp.join('..', 'outputs', 'baseline_densenet_svhn_20210102_114713'),
         osp.join('..', 'outputs', 'odin_densenet_svhn_20210102_152929'),
         osp.join('..', 'outputs', 'gram_densenet_svhn_20210102_115127')
         ],
        # Resnet SVHN
        [osp.join('..', 'outputs', 'baseline_resnet_svhn_20210102_115031'),
         osp.join('..', 'outputs', 'odin_resnet_svhn_20210102_152931'),
         osp.join('..', 'outputs', 'gram_resnet_svhn_20210102_115233')
         ]
    ]

    t0 = time.time()
    model_results = []
    for i, roots_i in enumerate(roots):

        method_dicts_i = {}
        method_i, model_name_i, trainset_name_i = '', '', ''
        for root_i in roots_i:
            method_i, model_name_i, trainset_name_i = osp.basename(root_i).split('_')[0:3]
            if method_i == 'baseline':
                pnml_df, max_prob_df = calc_performance_baseline(root_i, trainset_name_i)
            elif method_i == 'odin':
                pnml_df, max_prob_df = calc_performance_odin(root_i, trainset_name_i)
            elif method_i == 'gram':
                pnml_df, max_prob_df = calc_performance_gram(root_i, trainset_name_i)
            else:
                raise ValueError(f'{method_i} is not supported')

            method_dicts_i[method_i] = max_prob_df
            method_dicts_i[f'{method_i}+pNML'] = pnml_df

        # create nested df: metric -> method -> values
        df = create_nested_df(method_dicts_i)
        df['model_name'] = f'{model_name_i}_{trainset_name_i}'
        model_results.append(df)

        print(f'[{i}/{len(roots)-1}]')

    print('Finish in {:.2f} sec'.format(time.time() - t0))
    result_df = pd.concat(model_results, axis=0)
    result_df = result_df.set_index(["model_name", "ood_dataset"], inplace=False).round(1).dropna()
    print(result_df)
    result_df.to_csv('../outputs/results.csv')
