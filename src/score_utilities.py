import os.path as osp
import numpy as np
from tqdm import tqdm


def load_trainset_features(model_name: str, trainset_name: str, features_path: str):
    prefix = '{}_{}_{}_{}'.format(model_name, trainset_name, trainset_name, 'trainset')

    # Trainset load
    file_name = osp.join(features_path, '{}_features.npy'.format(prefix))
    trainset_features = np.load(file_name).T

    # Trainset labels load
    file_name = osp.join(features_path, '{}_train_labels.npy'.format(trainset_name))
    trainset_labels = np.load(file_name)

    return trainset_features, trainset_labels


def load_test_products(model_name: str, trainset_name: str, testset_name: str, features_path: str, logits_path: str):
    prefix = '{}_{}_{}_{}'.format(model_name, trainset_name, testset_name, 'testset')

    # Testset features load
    file_name = osp.join(features_path, '{}_features.npy'.format(prefix))
    testset_features = np.load(file_name).T

    # Testset outputs
    file_name = osp.join(logits_path, '{}_outputs.npy'.format(prefix))
    testset_outputs = np.load(file_name)
    testset_pred = testset_outputs.argmax(axis=1)

    return testset_features, testset_pred


def execute_svd_decomposition(x_m_train: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    u, s, vh = np.linalg.svd(x_m_train)
    eta = s ** 2

    # Pad to fit u
    padded = np.zeros(u.shape[0])
    padded[:eta.shape[0]] = eta
    eta = padded

    eta = eta[:, np.newaxis]

    return u, eta, vh


def decompose_trainset(trainset: np.ndarray, labels: np.ndarray):
    label_list = np.unique(labels).tolist()
    svd_list = []
    for num in tqdm(label_list):
        trainset_single_cls = trainset[:, labels == num]
        u, eta, vh = execute_svd_decomposition(trainset_single_cls)
        svd_list.append((u, eta, vh))
    return svd_list


def project_on_trainset(x_m_test: np.ndarray,
                        u: np.ndarray, eta: np.ndarray, vh: np.ndarray,
                        lamb: float = 1e-9) -> np.ndarray:
    n = len(vh)

    # x_m_test = x_m_test / np.linalg.norm(x_m_test, axis=0)  # keep?
    x_t_u_2 = (x_m_test.T.dot(u)) ** 2
    div = x_t_u_2 / (eta.T + lamb)

    distance = (1 / n) * div.sum(axis=1)
    return distance


def project_testset(testset: np.ndarray, test_pred: np.ndarray, svd_list: list) -> np.ndarray:
    distance_list = []
    for num, (u, eta, vh) in enumerate(svd_list):
        testset_single_cls = testset[:, test_pred == num]
        distance = project_on_trainset(testset_single_cls, u, eta, vh)
        distance_list.append(distance)

    distance_np = np.hstack(distance_list)
    return distance_np


def decompose_train_with_test(trainset: np.ndarray, labels: np.ndarray, testset: np.ndarray, test_pred: np.ndarray):
    label_list = np.unique(labels).tolist()
    svd_list = []
    for num in tqdm(label_list):
        trainset_single_cls = trainset[:, labels == num]
        testset_single_cls = testset[:, test_pred == num]
        for single in testset_single_cls:
            joint = np.hstack((trainset_single_cls, single))
            u, eta, vh = execute_svd_decomposition(joint)
            svd_list.append((u, eta, vh))
