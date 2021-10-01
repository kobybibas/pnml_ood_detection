import copy
import os

import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as npl
from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm as tqdm


# Numeric pNML
def add_to_test(x_train, y_train, x, y):
    return np.concatenate((x_train, x)), np.concatenate((y_train, y))


def fit_mlp(phi_train, y_train, initial_clf=None):
    if initial_clf is None:
        clf = MLPClassifier(
            solver="sgd",
            alpha=0.0,
            hidden_layer_sizes=(2,),
            random_state=0,
            max_iter=10000,
            activation="identity",
        ).fit(phi_train, y_train)
    else:
        clf = initial_clf.fit(phi_train, y_train)
    return clf


def calc_numerical_regret(
    phi_train: np.ndarray, y_train: np.ndarray, phi_test: np.ndarray
) -> np.ndarray:
    regret_list = []
    clf_erm = fit_mlp(phi_train, y_train)
    for phi in tqdm(phi_test):
        phi = np.expand_dims(phi, 0)

        phi_all, y_all = add_to_test(phi_train, y_train, phi, np.array([0]))
        clf = fit_mlp(phi_all, y_all, copy.deepcopy(clf_erm))
        p0 = clf.predict_proba(phi)[0][0]

        phi_all, y_all = add_to_test(phi_train, y_train, phi, np.array([1]))
        clf = fit_mlp(phi_all, y_all, copy.deepcopy(clf_erm))
        p1 = clf.predict_proba(phi)[0][1]

        regret = np.log2(p0 + p1)
        regret_list.append(regret)
    return np.asarray(regret_list)


def add_bias(x):
    n = x.shape[0]
    ones = np.expand_dims(np.ones(n), 1)
    return np.hstack((x, ones))


def create_testset(array_1, array_2):
    mesh = np.array(np.meshgrid(array_1, array_2))
    combinations = mesh.T.reshape(-1, 2)
    return add_bias(combinations)


def load_iris_dataset(is_seperable: bool = True):
    # import some data to play with
    iris = datasets.load_iris()
    x_train = iris.data[:, :2]  # we only take the first two features.
    x_train = add_bias(x_train)
    y_train = iris.target

    if is_seperable is True:
        # For seperable data, get only label 0 and 1
        x_train = x_train[y_train != 1]
        y_train = y_train[y_train != 1]
        y_train[y_train == 2] = 1

        # Reduce number of training set
        x_train = x_train[::6]
        y_train = y_train[::6]

        x1_min, x1_max = x_train[:, 0].min() - 1.0, x_train[:, 0].max() + 1.0
        x2_min, x2_max = x_train[:, 1].min() - 0.5, x_train[:, 1].max() + 0.5
    else:
        # For mixed data, get only label 1 and 2
        x_train = x_train[y_train != 0]
        y_train = y_train[y_train != 0]
        y_train[y_train == 2] = 0
        x_train = x_train[::5]
        y_train = y_train[::5]

        x1_min, x1_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 2
        x2_min, x2_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 2

    x_test = create_testset(
        np.linspace(x1_min, x1_max, 100), np.linspace(x2_min, x2_max, 100)
    )

    return x_train, y_train, x_test, x1_min, x1_max, x2_min, x2_max


def calc_prediction_and_regret(x_train, x_test, erm_probs):
    # Analytical pNML
    regrets = []
    x_t_gs = []
    pnml_probs = []
    n, m = x_train.shape

    X_inv = npl.pinv(x_train)
    P_bot = np.eye(m) - X_inv @ x_train

    for x, probs_i in zip(x_test, erm_probs):
        # Convert to column vec
        x = np.expand_dims(x, 1)
        x = x / npl.norm(x, keepdims=True)
        x_bot = P_bot @ x
        x_bot_square = float(x_bot.T @ x_bot) ** 2

        x_parallel = float(x.T @ X_inv @ X_inv.T @ x)
        if x_bot_square > np.finfo("float").eps:
            x_t_g = 1  # x_bot.T @ x_bot
        else:  # x_bot =0
            x_t_g = x_parallel / (1 + x_parallel)

        genies_i = probs_i / (probs_i + (1 - probs_i) * (probs_i ** x_t_g))
        nf = np.sum(genies_i)
        regret_i = float(np.log2(nf))
        regrets.append(regret_i)
        x_t_gs.append(x_t_g)
        pnml_probs.append(genies_i / nf)

    pnml_probs = np.asarray(pnml_probs).squeeze()
    x_t_gs = np.asarray(x_t_gs).squeeze()
    regrets = np.asarray(regrets).squeeze()
    return pnml_probs, regrets
