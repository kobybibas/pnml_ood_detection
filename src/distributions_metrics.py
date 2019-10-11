import numpy as np
import pandas as pd
from scipy.ndimage.filters import gaussian_filter
from sklearn.metrics import roc_auc_score, average_precision_score
import sklearn.metrics as metrics


def bhattacharyya(a, b):
    return -np.log(np.sum(np.sqrt(a * b)))


def kl(p, q):
    p = np.asarray(p, dtype=np.float)
    p[p == 0] = np.finfo(float).eps

    q = np.asarray(q, dtype=np.float)
    q[q == 0] = np.finfo(float).eps
    return np.sum(np.where(p > np.finfo(float).eps, - p * np.log(q / p), 0))


def kl_distance(a, b):
    return 0.5 * (kl(a, b) + kl(b, a))


def histogram_and_smooth(data, nbins=10, sigma=1):
    hist = np.histogram(data, bins=nbins)[0]
    hist_smooth = gaussian_filter(hist, sigma)
    hist_smooth_normalized = hist_smooth / np.sum(hist_smooth)
    return hist_smooth_normalized


def calc_p_lamb(a_distribution, b_distribution, lamb):
    """
    Calc P_lamb distribution: P_lambda = a^lamb * b^(1-lamb) / Normalize
    """
    p_lamb = np.power(a_distribution, lamb) * np.power(b_distribution, 1 - lamb)
    p_lamb /= np.sum(p_lamb)
    return p_lamb


def find_lamb_star(a_distribution, b_distribution):
    """
    Find lambda^* for which D(a||P_lambda) = D(b||P_lambda)
    """
    distance = []
    lamb_list = np.flip(np.linspace(0, 1, 20))

    # For each lambda calculate the Distance D(a||P_lambda) - D(b||P_lambda)
    for lamb in lamb_list:
        p_lamb = calc_p_lamb(a_distribution, b_distribution, lamb)
        distance.append(kl(p_lamb, a_distribution) -
                        kl(p_lamb, b_distribution))

    # Find lambda for which the the distance is zero
    lamb_star = np.interp(0.0, distance, lamb_list)

    # Get the KL value in that lamb_star: D(a||P_lambda)
    p_lamb_star = calc_p_lamb(a_distribution, b_distribution, lamb_star)
    return kl(p_lamb_star, a_distribution)


def calc_performance_in_out_dist(true_ind, score_ind):
    """
    Calculate evaluation matrics
    """
    score_ood = 1 - np.array(score_ind)
    true_ood = 1 - np.array(true_ind)

    true_len = np.sum(true_ind)
    false_len = len(true_ind) - true_len
    sample_weight = [1] * true_len + [true_len / false_len] * false_len

    # AUROC
    res_auc = roc_auc_score(true_ind, score_ind, sample_weight=sample_weight)

    # AUPR-Out
    ap_out = average_precision_score(
        true_ood, score_ood, sample_weight=sample_weight)

    # AUPR-In
    ap_in = average_precision_score(
        true_ind, score_ind, sample_weight=sample_weight)

    # # KL distance
    # score_ind = np.array(score_ind)
    # true_ind = np.array(true_ind)
    # in_hist = histogram_and_smooth(
    #     score_ind[true_ind == True], nbins=20, sigma=0)
    # out_hist = histogram_and_smooth(
    #     score_ind[true_ind == False], nbins=20, sigma=0)
    # kl_dist = kl_distance(in_hist, out_hist)
    #
    # # Bhattacharyya distance
    # bhatt_dist = bhattacharyya(in_hist, out_hist)
    #
    # # P_lambda
    # kl_in_p_lambda = find_lamb_star(in_hist, out_hist)

    # Find index of TPR = 95%
    fpr, tpr, thresholds = metrics.roc_curve(true_ind, score_ind)
    idx_tpr_95 = np.argmin(np.abs(tpr - 0.95))
    fpr_in_tpr_95 = fpr[idx_tpr_95]

    # Detection Error: when TPR is 95%.
    detection_error = 0.5 * (1 - 0.95) + 0.5 * fpr_in_tpr_95

    ood_df = pd.DataFrame({
        'FPR (95% TPR) ↓': [fpr_in_tpr_95 * 100],
        'Detection Error ↓': [detection_error * 100],
        'AUROC ↑': [res_auc * 100],
        'AP-In ↑': [ap_in * 100],
        'AP-Out ↑': [ap_out * 100],
        # 'KL Divergence': [kl_dist],
        # 'Bhattach Distance': [bhatt_dist],
        # 'KL in P_lamb': [kl_in_p_lambda]
    })

    return ood_df
