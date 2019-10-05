import copy
import os.path as osp

import numpy as np
import pandas as pd

from distributions_metrics import calc_performance_in_out_dist
from tqdm import tqdm


def compute_regret(x_m_train: np.ndarray, x_m_test: np.ndarray, lamb: float, u=None, eta=None, vh=None, dim: int = -1):
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
    div = x_t_u_2 / (eta.T + lamb)

    div = div[:, :dim]

    regret_list = np.log(1 + (1 / n) * div.sum(axis=1)).tolist()
    return regret_list, u, eta, vh


def analyze_ind_ood_regret(trainset_all: np.ndarray, train_labels: np.ndarray, testset_ind: np.ndarray,
                           testset_ood: np.ndarray, dim: int = -1):
    regret_ind_list_all = []
    regret_ood_list_all = []
    lamb = 1e-16
    for class_num in np.unique(train_labels):
        trainset = trainset_all[:, train_labels == class_num]
        regret_ind_list, u, eta, vh = compute_regret(trainset, testset_ind, lamb=lamb, u=None, eta=None, vh=None,
                                                     dim=dim)
        regret_ood_list, _, _, _ = compute_regret(trainset, testset_ood, lamb=lamb, u=u, eta=eta, vh=vh, dim=dim)

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


def optimize_dim_with_range(trainset_all, train_labels, testset_ind, testset_ood,
                            start: int = 10, stop=None, step: int = 100):
    score_max = 0
    dim_max = 0
    score_list = []
    if stop is None:
        stop = trainset_all.shape[0]
    dim_np = np.arange(start, stop, step, dtype=int)
    dim_np = np.append(dim_np, stop)
    print('Iterating over: ', dim_np)
    for dim in tqdm(dim_np):
        perf_regret, regret_ind_list, regret_ood_list = analyze_ind_ood_regret(trainset_all, train_labels,
                                                                               testset_ind,
                                                                               testset_ood,
                                                                               dim=dim)
        score = float(perf_regret['AUROC ↑'])
        score_list.append(score)
        if score > score_max:
            score_max = score
            dim_max = int(dim)
            # print('Found max. [dim AUROC]=[{} {:.2f}]'.format(dim, score))
    return dim_max, dim_np, score_list


def optimize_dim(trainset_path, train_labels_path, testset_ind_path, testset_ood_path):
    trainset_all = np.load(trainset_path).T
    testset_ind = np.load(testset_ind_path).T
    train_labels = np.load(train_labels_path)
    testset_ood = np.load(testset_ood_path).T

    dim_max = trainset_all.shape[0]
    start = 10
    stop = None
    for step in [100, 10, 1]:
        dim_max, dim_np, score_list = optimize_dim_with_range(trainset_all, train_labels, testset_ind,
                                                              testset_ood,
                                                              start=start, stop=stop, step=step)
        assert isinstance(dim_max, int)
        assert isinstance(dim_np, np.ndarray)
        assert isinstance(score_list, list)

        # Refine
        index_min = np.argmin(np.abs(dim_np - dim_max))
        start = dim_np[index_min - 1] if index_min != 0 else dim_np[index_min]
        stop = dim_np[index_min + 1] if index_min != len(dim_np) - 1 else dim_np[index_min]
        print('[AUROC dim]=[{:.2f} {}] [start end stop]=[{} {} {}]'.format(score_list[index_min],
                                                                           dim_max, start, stop, step))

    return dim_max


def benchmark_pnml(pnml_dataset_input: dict, dim: int = -1):
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
                                                                               testset_ood,
                                                                               dim=dim)
        # perf_regret = perf_regret.rename(index={0: ood_name})
        perf_regret.insert(0, 'Method', ['pNML'])
        perf_regret.insert(0, 'OOD Datasets', [ood_name])

        perf_df_list.append(perf_regret)

    df = pd.concat(perf_df_list)
    df.index = pd.RangeIndex(len(df.index))
    return df


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


# ------ #
# ODIN
# ------ #
datasets_odin = ['Imagenet (crop)', 'Imagenet (resize)', 'LSUN (crop)',
                 'LSUN (resize)', 'iSUN', 'Uniform', 'Gaussian']
densnet_cifar10_odin = {
    'Model': ['DensNet-BC CIFAR10'] * len(datasets_odin),
    'OOD Datasets': datasets_odin,
    'Method': ['ODIN'] * len(datasets_odin),
    'FPR (95% TPR) ↓': [4.3, 7.5, 8.7, 3.8, 6.3, 0.0, 0.0],
    'Detection Error ↓': [4.7, 6.3, 6.9, 4.4, 5.7, 2.5, 2.5],
    'AUROC ↑': [99.1, 98.5, 98.2, 99.2, 98.8, 99.9, 100.0],
    'AP-In ↑': [99.1, 98.6, 98.5, 99.3, 98.9, 100.0, 100.0],
    'AP-Out ↑': [99.1, 98.5, 97.8, 99.2, 98.8, 99.9, 100.0],
}
odin_df_densnet_cifar10 = pd.DataFrame(densnet_cifar10_odin)

densnet_cifar100_odin = {
    'Model': ['DensNet-BC CIFAR100'] * len(datasets_odin),
    'OOD Datasets': datasets_odin,
    'Method': ['ODIN'] * len(datasets_odin),
    'FPR (95% TPR) ↓': [17.3, 44.3, 17.6, 44.0, 49.5, 0.5, 0.2],
    'Detection Error ↓': [11.2, 24.6, 11.3, 24.5, 27.2, 2.8, 2.6],
    'AUROC ↑': [97.1, 90.7, 96.8, 91.5, 90.1, 99.5, 99.6],
    'AP-In ↑': [97.4, 91.4, 97.1, 92.4, 91.1, 99.6, 99.7],
    'AP-Out ↑': [96.8, 90.1, 96.5, 90.6, 88.9, 99.0, 99.1]}
odin_df_densnet_cifar100 = pd.DataFrame(densnet_cifar100_odin)

resnet_cifar10_odin = {
    'Model': ['WRN-28-10 CIFAR10'] * len(datasets_odin),
    'OOD Datasets': datasets_odin,
    'Method': ['ODIN'] * len(datasets_odin),
    'FPR (95% TPR) ↓': [23.4, 25.5, 21.8, 17.6, 21.3, 0.0, 0.0],
    'Detection Error ↓': [14.2, 15.2, 13.4, 11.3, 13.2, 2.5, 2.5],
    'AUROC ↑': [94.2, 92.1, 95.9, 95.4, 93.7, 100.0, 100.0],
    'AP-In ↑': [92.8, 89.0, 95.8, 93.8, 91.2, 100.0, 100.0],
    'AP-Out ↑': [94.7, 93.6, 95.5, 96.1, 94.9, 100.0, 100.0]}
odin_df_resnet_cifar10 = pd.DataFrame(resnet_cifar10_odin)

resnet_cifar100_odin = {
    'Model': ['WRN-28-10 CIFAR100'] * len(datasets_odin),
    'OOD Datasets': datasets_odin,
    'Method': ['ODIN'] * len(datasets_odin),
    'FPR (95% TPR) ↓': [43.9, 55.9, 39.6, 56.5, 57.3, 0.1, 1.0],
    'Detection Error ↓': [24.4, 30.4, 22.3, 30.8, 31.1, 2.5, 3.0],
    'AUROC ↑': [90.8, 84.0, 92.0, 86.0, 85.6, 99.1, 98.5],
    'AP-In ↑': [91.4, 82.8, 92.4, 86.2, 85.9, 99.4, 99.1],
    'AP-Out ↑': [90.0, 84.4, 91.6, 84.9, 84.8, 97.5, 95.9]}
odin_df_resnet_cifar100 = pd.DataFrame(resnet_cifar100_odin)

# ------------- #
# Leave one out
# ------------- #
datasets_loo = ['Imagenet (crop)', 'Imagenet (resize)', 'LSUN (crop)', 'LSUN (resize)', 'Uniform', 'Gaussian']
densnet_cifar10_lvo = {
    'Model': ['DensNet-BC CIFAR10'] * len(datasets_loo),
    'OOD Datasets': datasets_loo,
    'Method': ['LOO'] * len(datasets_loo),
    'FPR (95% TPR) ↓': [1.23, 2.93, 3.42, 0.77, 2.61, 0.00],
    'Detection Error ↓': [2.63, 3.84, 4.12, 2.1, 3.6, 0.2],
    'AUROC ↑': [99.65, 99.34, 99.25, 99.75, 98.55, 99.84],
    'AP-In ↑': [99.68, 99.37, 99.29, 99.77, 98.94, 99.86],
    'AP-Out ↑': [99.64, 99.32, 99.24, 99.73, 97.52, 99.6]
}
lvo_df_densnet_cifar10 = pd.DataFrame(densnet_cifar10_lvo)

densnet_cifar100_lvo = {
    'Model': ['DensNet-BC CIFAR100'] * len(datasets_loo),
    'OOD Datasets': datasets_loo,
    'Method': ['LOO'] * len(datasets_loo),
    'FPR (95% TPR) ↓': [8.29, 20.52, 14.69, 16.23, 79.73, 38.52],
    'Detection Error ↓': [6.27, 9.98, 8.46, 8.77, 9.46, 8.21],
    'AUROC ↑': [98.43, 96.27, 97.37, 97.03, 92.0, 94.89],
    'AP-In ↑': [98.58, 96.66, 97.62, 97.37, 94.77, 96.36],
    'AP-Out ↑': [98.3, 95.82, 97.18, 96.6, 83.81, 90.01]}
lvo_df_densnet_cifar100 = pd.DataFrame(densnet_cifar100_lvo)

resnet_cifar10_lvo = {
    'Model': ['WRN-28-10 CIFAR10'] * len(datasets_loo),
    'OOD Datasets': datasets_loo,
    'Method': ['LOO'] * len(datasets_loo),
    'FPR (95% TPR) ↓': [0.82, 2.94, 1.93, 0.88, 16.39, 0.0],
    'Detection Error ↓': [2.24, 3.83, 3.24, 2.52, 5.39, 1.03],
    'AUROC ↑': [99.75, 99.36, 99.55, 99.7, 96.77, 99.58],
    'AP-In ↑': [99.77, 99.4, 99.57, 99.72, 97.78, 99.71],
    'AP-Out ↑': [99.75, 99.36, 99.55, 99.68, 94.18, 99.2]}
lvo_df_resnet_cifar10 = pd.DataFrame(resnet_cifar10_lvo)

resnet_cifar100_lvo = {
    'Model': ['WRN-28-10 CIFAR10'] * len(datasets_loo),
    'OOD Datasets': datasets_loo,
    'Method': ['LOO'] * len(datasets_loo),
    'FPR (95% TPR) ↓': [9.17, 24.53, 14.22, 16.53, 99.9, 98.26],
    'Detection Error ↓': [6.67, 11.64, 8.2, 9.14, 14.86, 16.88],
    'AUROC ↑': [98.22, 95.18, 97.38, 96.77, 83.44, 93.04],
    'AP-In ↑': [98.39, 95.5, 97.62, 97.03, 89.43, 88.64],
    'AP-Out ↑': [98.07, 94.78, 97.16, 96.41, 71.2, 71.62]}
lvo_df_resnet_cifar100 = pd.DataFrame(resnet_cifar100_lvo)
