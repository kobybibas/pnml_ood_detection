import os.path as osp

import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm

from distributions_metrics import calc_performance_in_out_dist
from ood_score_utilities import compute_regret, calc_loss, sigmoid, compute_regret_from_2_classes

model_name_map = {
    'DensNet-BC CIFAR10': 'densenet_cifar10',
    'DensNet-BC CIFAR100': 'densenet_cifar100',
    'WRN-28-10 CIFAR10': 'resnet_cifar10',
    'WRN-28-10 CIFAR100': 'resnet_cifar100'

}

testset_name_map = {
    'Imagenet (crop)': 'Imagenet',
    'Imagenet (resize)': 'Imagenet_resize',
    'LSUN (crop)': 'LSUN',
    'LSUN (resize)': 'LSUN_resize',
    'iSUN': 'iSUN',
    'Uniform': 'Uniform',
    'Gaussian': 'Gaussian',
}


class BenchmarkPNML:
    def __init__(self, base_dir_path=osp.join('..', 'output', 'embedding'),
                 is_5_aug: bool = False,
                 is_genie_estimate: bool = False):
        self.base_dir = base_dir_path
        self.is_5_aug = is_5_aug
        self.is_genie_estimate = is_genie_estimate
        self.prob_debug_dict = None
        self.regret_debug_dict = None

    def load_results_for_max_prob(self, model_name: str, testset_name: str) -> (

            np.ndarray, np.ndarray, np.ndarray):
        # Testset IND
        trainset_name = model_name.split(' ')[-1].lower()
        testset_output_ind_path = osp.join(self.base_dir,
                                           model_name_map[
                                               model_name] + '_' + trainset_name + '_' + 'test_outputs_pnml.npy')
        testset_output_ind = np.load(testset_output_ind_path)

        # Testset OOD
        testset_output_ood_path = osp.join(self.base_dir, model_name_map[model_name] + '_' + testset_name_map[
            testset_name] + '_' + 'test_outputs_pnml.npy')
        testset_output_ood = np.load(testset_output_ood_path)

        # Trainset labels load
        trainset_labels_path = osp.join(self.base_dir, trainset_name + '_' + 'train_labels.npy')
        trainset_labels = np.load(trainset_labels_path)
        return trainset_labels, testset_output_ind, testset_output_ood

    def calc_max_prob_score(self, model_name, ood_name_list):
        max_prob_score_dict = {}
        max_prob_debug_dict = {}
        for ood_name in ood_name_list:
            trainset_labels, testset_output_ind, testset_output_ood = self.load_results_for_max_prob(
                model_name,
                ood_name)
            num_labels = len(np.unique(trainset_labels))
            score_ind = calc_loss(testset_output_ind, num_labels)
            score_ood = calc_loss(testset_output_ood, num_labels)

            max_prob_score_dict[ood_name] = {'IND': score_ind.tolist(),
                                             'OOD': score_ood.tolist()}

            max_prob_debug_dict[ood_name] = {'IND': sigmoid(testset_output_ind),
                                             'OOD': sigmoid(testset_output_ood)}

        return max_prob_score_dict, max_prob_debug_dict

    def load_results_for_regret(self, model_name: str, testset_name: str) -> (np.ndarray,
                                                                              np.ndarray,
                                                                              np.ndarray,
                                                                              np.ndarray):
        aug_num = 1

        # Trainset load
        trainset_name = model_name.split(' ')[-1].lower()
        trainset_path = osp.join(self.base_dir,
                                 model_name_map[model_name] + '_' + trainset_name + '_' + 'train_pnml.npy')
        trainset = np.load(trainset_path).T

        if self.is_5_aug is True:
            trainset_list = []
            for i in range(aug_num):
                trainset_path = osp.join(self.base_dir,
                                         model_name_map[
                                             model_name] + '_' + trainset_name + '_' + 'train_pnml_%d.npy' % i)
                trainset_list.append(np.load(trainset_path).T)
            trainset = np.vstack([trainset] + trainset_list)

        # Trainset labels load
        trainset_labels_path = osp.join(self.base_dir, trainset_name + '_' + 'train_labels.npy')
        trainset_labels = np.load(trainset_labels_path)

        # self.is_5_aug is True:
        #     trainset_labels_list = []
        #     for i in range(aug_num):
        #         trainset_labels_list.append(np.load(trainset_labels_path).T)
        #     trainset_labels = np.hstack([trainset_labels] + trainset_labels_list)

        # Testset In-Dist load
        testset_ind_path = osp.join(self.base_dir,
                                    model_name_map[model_name] + '_' + trainset_name + '_' + 'test_pnml.npy')
        testset_ind = np.load(testset_ind_path).T

        if self.is_5_aug is True:
            testset_ind_list = []
            for i in range(aug_num):
                testset_ind_path = osp.join(self.base_dir,
                                            model_name_map[
                                                model_name] + '_' + trainset_name + '_' + 'test_pnml_%d.npy' % i)
                testset_ind_list.append(np.load(testset_ind_path).T)
            testset_ind = np.vstack([testset_ind] + testset_ind_list)

        # Testset OOD
        testset_ood_path = osp.join(self.base_dir, model_name_map[model_name] + '_' + testset_name_map[
            testset_name] + '_' + 'test_pnml.npy')
        testset_ood = np.load(testset_ood_path).T

        if self.is_5_aug is True:
            testset_ood_list = []
            for i in range(aug_num):
                testset_ood_path = osp.join(self.base_dir,
                                            model_name_map[model_name] + '_' + testset_name_map[
                                                testset_name] + '_' + 'test_pnml_%d.npy' % i)
                testset_ood_list.append(np.load(testset_ood_path).T)
            testset_ood = np.vstack([testset_ood] + testset_ood_list)

        return trainset, trainset_labels, testset_ind, testset_ood

    def calc_regret_score(self, model_name: str, ood_name_list: list) -> (dict, dict):
        trainset_decomp = None
        regret_score_dict = {}
        regret_debug_dict = {}
        for ood_name in tqdm(ood_name_list):
            print('regret score for {}'.format(ood_name))
            trainset, trainset_labels, testset_ind, testset_ood = self.load_results_for_regret(model_name, ood_name)
            regret_ind_np, regret_ood_np, trainset_decomp = compute_regret(trainset, trainset_labels,
                                                                           testset_ind, testset_ood,
                                                                           lamb=1e-9,
                                                                           trainset_decomp=trainset_decomp)

            regret_ind_list = np.min(regret_ind_np, axis=1).tolist()
            regret_ood_list = np.min(regret_ood_np, axis=1).tolist()

            regret_score_dict[ood_name] = {'IND': regret_ind_list,
                                           'OOD': regret_ood_list}

            regret_debug_dict[ood_name] = {'IND': regret_ind_np,
                                           'OOD': regret_ood_np}

        return regret_score_dict, regret_debug_dict

    def compute_perf_single(self, model_name: str, method: str, ood_name: str, regret_score_dict: dict,
                            loss_score_dict: dict):
        regret_ind_score_list, regret_ood_score_list = regret_score_dict['IND'], regret_score_dict['OOD']

        score_ind = regret_ind_score_list
        score_ood = regret_ood_score_list
        if self.is_genie_estimate is True:
            loss_ind_score_list, loss_ood_score_list = loss_score_dict['IND'], loss_score_dict['OOD']
            score_ind = (np.array(loss_ind_score_list) - np.array(regret_ind_score_list)).tolist()
            score_ood = (np.array(loss_ood_score_list) - np.array(regret_ood_score_list)).tolist()

        assert isinstance(score_ind, list) and isinstance(score_ood, list)

        y_score_ind = score_ind + score_ood
        y_score_ind = 1 - np.array(y_score_ind)
        y_true_ind = [True] * len(score_ind) + [False] * len(score_ood)
        perf_regret = calc_performance_in_out_dist(y_true_ind, y_score_ind)

        perf_regret.insert(0, 'Method', [method])
        perf_regret.insert(0, 'OOD Datasets', [ood_name])
        perf_regret.insert(0, 'Model', [model_name] * len(perf_regret))

        return perf_regret

    def execute(self, ood_name_list: list, model_name: str, method: str) -> (pd.DataFrame, dict):
        print('benchmark_pnml. model_name={}, is_5_aug={}, is_genie_estimate={}'.format(model_name,
                                                                                        self.is_5_aug,
                                                                                        self.is_genie_estimate))
        loss_score_dict, self.prob_debug_dict = self.calc_max_prob_score(model_name, ood_name_list)
        regret_score_dict, self.regret_debug_dict = self.calc_regret_score(model_name, ood_name_list)

        perf_df_list = []
        for (ood_regret_name, regret_score_dict), (ood_loss_name, loss_score_dict) in zip(regret_score_dict.items(),
                                                                                          loss_score_dict.items()):
            assert ood_regret_name == ood_loss_name
            perf_regret = self.compute_perf_single(model_name, method, ood_regret_name, regret_score_dict,
                                                   loss_score_dict)
            perf_df_list.append(perf_regret)

        df = pd.concat(perf_df_list)
        df.index = pd.RangeIndex(len(df.index))

        debug_dict = {'regret': self.regret_debug_dict, 'prob': self.prob_debug_dict}
        return df, debug_dict


class BenchmarkPNML2Regrets(BenchmarkPNML):
    def compose_pair(self, outputs):
        pair_list = []
        for output_single in outputs:
            # label1 = output_single.argmax()
            # label2 = output_single.argmin()

            indexes = np.argsort(-output_single)
            label1, label2 = indexes[0], indexes[1]

            pair_single = [label1, label2]
            pair_single.sort()
            pair_list.append(tuple(pair_single))

        return pair_list

    def execute(self, ood_name_list: list, model_name: str, method: str) -> (pd.DataFrame, dict):
        print('benchmark_pnml. model_name={}, is_5_aug={}, is_genie_estimate={}'.format(model_name,
                                                                                        self.is_5_aug,
                                                                                        self.is_genie_estimate))
        loss_score_dict, self.prob_debug_dict = self.calc_max_prob_score(model_name, ood_name_list)

        perf_df_list = []
        trainset_decomp = None
        for ood_name in ood_name_list:
            # Compose pair
            print(ood_name)
            trainset_labels, testset_output_ind, testset_output_ood = self.load_results_for_max_prob(model_name,
                                                                                                     ood_name)
            output_ind_pairs = self.compose_pair(testset_output_ind)
            output_ood_pairs = self.compose_pair(testset_output_ood)

            # Load regret
            trainset, trainset_labels, testset_ind, testset_ood = self.load_results_for_regret(model_name, ood_name)

            # Compute regret
            regret_ind_np, regret_ood_np, trainset_decomp = compute_regret_from_2_classes(trainset, trainset_labels,
                                                                                          testset_ind, testset_ood,
                                                                                          output_ind_pairs,
                                                                                          output_ood_pairs,
                                                                                          trainset_decomp)
            regret_score_dict = {'IND': regret_ind_np.tolist(), 'OOD': regret_ood_np.tolist()}
            perf_regret = self.compute_perf_single(model_name, method, ood_name, regret_score_dict, {})
            perf_df_list.append(perf_regret)

        df = pd.concat(perf_df_list)
        df.index = pd.RangeIndex(len(df.index))

        debug_dict = {'regret': self.regret_debug_dict, 'prob': self.prob_debug_dict}
        return df, debug_dict


def benchmark_pnml(ood_name_list: list, model_name: str, method: str,
                   base_dir_path=osp.join('..', 'output', 'embedding'),
                   is_5_aug: bool = False,
                   is_genie_estimate: bool = False) -> (pd.DataFrame, dict):
    # benchmark_h = BenchmarkPNML(base_dir_path, is_5_aug, is_genie_estimate)
    benchmark_h = BenchmarkPNML2Regrets(base_dir_path, is_5_aug, is_genie_estimate)
    df, debug_dict = benchmark_h.execute(ood_name_list, model_name, method)
    return df, debug_dict


def cat_benchmark_df(odin_df: pd.DataFrame,
                     lvo_df: pd.DataFrame,
                     pnml_df: pd.DataFrame,
                     pnml_5_aug_df: pd.DataFrame,
                     ) -> pd.DataFrame:
    odin_df_dropped = odin_df[odin_df['OOD Datasets'] != 'iSUN']
    odin_df_dropped.index = pd.RangeIndex(len(odin_df_dropped.index))

    pnml_df_dropped = pnml_df[pnml_df['OOD Datasets'] != 'iSUN']
    pnml_df_dropped.index = pd.RangeIndex(len(pnml_df_dropped.index))

    pnml_5_aug_df_dropped = pnml_5_aug_df[pnml_5_aug_df['OOD Datasets'] != 'iSUN']
    pnml_5_aug_df_dropped.index = pd.RangeIndex(len(pnml_5_aug_df_dropped.index))

    merge_df = pd.concat([odin_df_dropped,
                          lvo_df,
                          pnml_df_dropped,
                          pnml_5_aug_df_dropped
                          ]).sort_index(kind='merge')

    return merge_df.round(1)


if __name__ == '__main__':
    from tqdm import tqdm as tqdm

    base_dir = osp.join('..', 'output', 'embedding')

    ood_name_benchmark_list = ['Imagenet (crop)', 'Imagenet (resize)', 'LSUN (crop)', 'LSUN (resize)',
                               'iSUN', 'Uniform', 'Gaussian']

    pnml_df_resnet_cifar100, _ = benchmark_pnml(ood_name_benchmark_list, 'DensNet-BC CIFAR100', 'pNML', base_dir,
                                                is_5_aug=False,
                                                is_genie_estimate=False)
    print(pnml_df_resnet_cifar100[['AUROC ↑', 'AP-In ↑']])
