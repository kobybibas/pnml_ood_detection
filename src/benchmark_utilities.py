import copy
import os.path as osp

import numpy as np
import pandas as pd

from distributions_metrics import calc_performance_in_out_dist


def compute_regret(x_m_train: np.ndarray, x_m_test: np.ndarray, lamb: float, u=None, eta=None, vh=None):
    n = x_m_train.shape[1]

    # SVD decomposition
    if u is None or eta is None or None or vh is None:
        u, s, vh = np.linalg.svd(x_m_train)
        eta = s ** 2

        # pad to fit u
        padded = np.zeros(u.shape[0])
        padded[:eta.shape[0]] = eta
        eta = padded

        eta = eta[:, np.newaxis]

    # Calc mean regret
    x_t_u_2 = (x_m_test.T.dot(u)) ** 2
    div = x_t_u_2 / \
          (eta +
           lamb) if eta.shape[0] == x_t_u_2.shape[0] else x_t_u_2 / (eta.T + lamb)
    regret_list = np.log(1 + (1 / n) * div.sum(axis=1)).tolist()
    return regret_list, u, eta, vh


def analyze_ind_ood_regret(trainset_all: np.ndarray, train_labels: np.ndarray, testset_ind: np.ndarray,
                           testset_ood: np.ndarray):
    regret_ind_list_all = []
    regret_ood_list_all = []
    lamb = 0
    for class_num in np.unique(train_labels):
        trainset = trainset_all[:, train_labels == class_num]
        regret_ind_list, u, eta, vh = compute_regret(trainset, testset_ind, lamb=lamb)
        regret_ood_list, _, _, _ = compute_regret(trainset, testset_ood, lamb=lamb, u=u, eta=eta, vh=vh)

        regret_ind_list_all.append(regret_ind_list)
        regret_ood_list_all.append(regret_ood_list)

    regret_ind_np = np.asarray(regret_ind_list_all)
    regret_ood_np = np.asarray(regret_ood_list_all)

    regret_ind_list = np.min(regret_ind_np, axis=0).tolist()
    regret_ood_list = np.min(regret_ood_np, axis=0).tolist()

    assert isinstance(regret_ind_list, list)
    assert isinstance(regret_ood_list, list)

    y_score_ind = regret_ind_list + regret_ood_list
    y_score_ind = 1 - np.array(y_score_ind)
    y_true_ind = [True] * len(regret_ind_list) + [False] * len(regret_ood_list)
    performance = calc_performance_in_out_dist(y_true_ind, y_score_ind)

    return performance, regret_ind_list, regret_ood_list


def benchmark_pnml(pnml_dataset_input: dict):
    pnml_dataset = copy.deepcopy(pnml_dataset_input)

    # Get In-Distribution datasets
    trainset_path = pnml_dataset.pop('trainset', None)
    train_labels_path = pnml_dataset.pop('train_labels', None)
    testset_path = pnml_dataset.pop('testset', None)
    trainset_all = np.load(trainset_path).T
    testset_ind = np.load(testset_path).T
    train_labels = np.load(train_labels_path)

    # Initialize performance dataframe output
    perf_df_list = []

    # Iterate over Out-Of-Distribution datasets
    for ood_name, npy_path in pnml_dataset.items():
        if osp.exists(npy_path) is False:
            continue
        print('pNML for {}'.format(ood_name))
        testset_ood = np.load(npy_path).T

        # Regret
        perf_regret, regret_ind_list, regret_ood_list = analyze_ind_ood_regret(trainset_all, train_labels,
                                                                               testset_ind,
                                                                               testset_ood)
        perf_regret = perf_regret.rename(index={0: ood_name})
        perf_regret.insert(0, 'Method', ['pNML'])

        perf_df_list.append(perf_regret)

    return pd.concat(perf_df_list)


def benchmark_odin(odin_dataset_input: dict) -> pd.DataFrame:
    odin_dataset = copy.deepcopy(odin_dataset_input)

    # Get In-Distribution datasets
    testset_path = odin_dataset.pop('testset', None)
    assert osp.exists(testset_path)
    ind_max_prob = np.load(testset_path).tolist()
    perf_df_list = []
    for ood_name, npy_path in odin_dataset.items():
        if osp.exists(npy_path) is False:
            continue
        print('ODIN for {}'.format(ood_name))
        out_dist_max_prob = np.load(npy_path).tolist()

        assert isinstance(ind_max_prob, list)
        assert isinstance(out_dist_max_prob, list)

        y_score_ind = ind_max_prob + out_dist_max_prob
        y_true_ind = [True] * len(ind_max_prob) + [False] * len(out_dist_max_prob)
        performance_odin = calc_performance_in_out_dist(y_true_ind, y_score_ind)

        performance_odin = performance_odin.rename(index={0: ood_name})
        performance_odin.insert(0, 'Method', ['ODIN'])
        perf_df_list.append(performance_odin)

    return pd.concat(perf_df_list)
