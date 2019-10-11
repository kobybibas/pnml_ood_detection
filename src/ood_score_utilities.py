import numpy as np
from tqdm import tqdm


def sigmoid(x, derivative=False):
    return x * (1 - x) if derivative else 1 / (1 + np.exp(-x))


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - x.max(axis=1)[:, np.newaxis])
    return e_x / e_x.sum(axis=1)[:, np.newaxis]


def decompose_trainset(x_m_train: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    u, s, vh = np.linalg.svd(x_m_train)
    eta = s ** 2

    # Pad to fit u
    padded = np.zeros(u.shape[0])
    padded[:eta.shape[0]] = eta
    eta = padded

    eta = eta[:, np.newaxis]

    return u, eta, vh


def project_on_trainset(x_m_test: np.ndarray, u: np.ndarray, eta: np.ndarray, vh: np.ndarray,
                        lamb: float = 0, log_base: float = np.e):
    n = len(vh)

    x_t_u_2 = (x_m_test.T.dot(u)) ** 2
    div = x_t_u_2 / (eta.T + lamb)

    regret_np = np.log(1 + (1 / n) * div.sum(axis=1)) / np.log(log_base)
    return regret_np


def compute_regret_ind_and_ood(trainset, train_labels, testset_ind, testset_ood,
                               output_ind_pairs: list,
                               output_ood_pairs: list,
                               trainset_decomp=None,
                               lamb: float = 1e-9
                               ):
    if trainset_decomp is None:
        trainset_decomp = {}

    unique_pairs = np.unique(output_ind_pairs + output_ood_pairs, axis=0)
    for label1, label2 in tqdm(unique_pairs):
        key_name = '%d %d' % (label1, label2)
        if key_name not in trainset_decomp:
            trainset_single_class = trainset[:, np.logical_or(train_labels == label1, train_labels == label2)]
            u, eta, vh = decompose_trainset(trainset_single_class)
            trainset_decomp[key_name] = (u, eta, vh)

    regret_ind_list = []
    regret_ood_list = []
    # log_base = len(np.unique(train_labels))
    log_base = 2

    for regret_list, output_pairs, testset in [(regret_ind_list, output_ind_pairs, testset_ind.T),
                                               (regret_ood_list, output_ood_pairs, testset_ood.T)]:
        for (label1, label2), test_single in zip(output_pairs, testset):
            key_name = '%d %d' % (label1, label2)
            u, eta, vh = trainset_decomp[key_name]

            regret = project_on_trainset(test_single[:, np.newaxis],
                                         u=u, eta=eta, vh=vh,
                                         lamb=lamb, log_base=log_base)

            regret_list.append(float(regret))

    regret_ind_np = np.array(regret_ind_list)
    regret_ood_np = np.array(regret_ood_list)

    # print('[u.shape, eta,shape, vh.shape]=[{} {} {}]'.format(u.shape, eta.shape, vh.shape))
    # print('[regret_ind_np.shape, regret_ood_np,shape]=[{} {}]'.format(regret_ind_np.shape, regret_ood_np.shape))

    return regret_ind_np, regret_ood_np, trainset_decomp


def calc_loss(output_np: np.ndarray, num_labels: int) -> np.ndarray:
    prob = softmax(output_np)
    logloss = -np.log(prob) / np.log(num_labels)
    return logloss.min(axis=1)
