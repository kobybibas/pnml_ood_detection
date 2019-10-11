import os.path as osp

import numpy as np
import pandas as pd

from distributions_metrics import calc_performance_in_out_dist
from ood_score_utilities import compute_regret_ind_and_ood

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
    def __init__(self, base_dir_path=osp.join('..', 'output', 'embedding')):
        self.base_dir = base_dir_path
        self.output_debug_dict = None
        self.regret_debug_dict = None
        self.method = 'pNML'

    def load_model_outputs(self, model_name: str, testset_name: str) -> (np.ndarray, np.ndarray, np.ndarray):
        trainset_name = model_name.split(' ')[-1].lower()
        testset_name_file = testset_name_map[testset_name]
        model_file_name = model_name_map[model_name]

        # Testset IND
        testset_output_ind_path = osp.join(self.base_dir,
                                           model_file_name + '_' + trainset_name + '_' + 'test_outputs_pnml.npy')
        testset_output_ind = np.load(testset_output_ind_path)

        # Testset OOD
        testset_output_ood_path = osp.join(self.base_dir,
                                           model_file_name + '_' + testset_name_file + '_' + 'test_outputs_pnml.npy')
        testset_output_ood = np.load(testset_output_ood_path)

        # Trainset labels load
        trainset_labels_path = osp.join(self.base_dir, trainset_name + '_' + 'train_labels.npy')
        trainset_labels = np.load(trainset_labels_path)
        return trainset_labels, testset_output_ind, testset_output_ood

    def load_model_embedding(self, model_name: str, testset_name: str) -> (np.ndarray, np.ndarray,
                                                                           np.ndarray, np.ndarray):
        trainset_name = model_name.split(' ')[-1].lower()
        testset_name_file = testset_name_map[testset_name]
        model_file_name = model_name_map[model_name]

        # Trainset load
        trainset_path = osp.join(self.base_dir, model_file_name + '_' + trainset_name + '_' + 'train_pnml.npy')
        trainset = np.load(trainset_path).T

        # Trainset labels load
        trainset_labels_path = osp.join(self.base_dir, trainset_name + '_' + 'train_labels.npy')
        trainset_labels = np.load(trainset_labels_path)

        # Testset In-Dist load
        testset_ind_path = osp.join(self.base_dir, model_file_name + '_' + trainset_name + '_' + 'test_pnml.npy')
        testset_ind = np.load(testset_ind_path).T

        # Testset OOD
        testset_ood_path = osp.join(self.base_dir, model_file_name + '_' + testset_name_file + '_' + 'test_pnml.npy')
        testset_ood = np.load(testset_ood_path).T

        return trainset, trainset_labels, testset_ind, testset_ood

    def compute_perf_single(self, model_name: str, ood_name: str, score_ind: list, score_ood: list):
        assert isinstance(score_ind, list) and isinstance(score_ood, list)

        y_score_ind = score_ind + score_ood
        y_score_ind = 1 - np.array(y_score_ind)
        y_true_ind = [True] * len(score_ind) + [False] * len(score_ood)
        perf_regret = calc_performance_in_out_dist(y_true_ind, y_score_ind)

        perf_regret.insert(0, 'Method', [self.method])
        perf_regret.insert(0, 'OOD Datasets', [ood_name])
        perf_regret.insert(0, 'Model', [model_name] * len(perf_regret))

        return perf_regret

    def compose_pair(self, outputs, first_top: int = 0, second_top: int = 1):
        indexes = np.argsort(-outputs, axis=1)
        labels_min_list = np.minimum(indexes[:, first_top], indexes[:, second_top])
        labels_max_list = np.maximum(indexes[:, first_top], indexes[:, second_top])
        pair_list = []
        for label_min, label_max in zip(labels_min_list, labels_max_list):
            pair_list.append((label_min, label_max))
        return pair_list

    def execute(self, ood_name_list: list, model_name: str) -> (pd.DataFrame, dict):
        print('benchmark_pnml. model_name={}'.format(model_name))

        perf_df_list = []
        trainset_decomp = None
        self.output_debug_dict = {}
        self.regret_debug_dict = {}
        for ood_name in ood_name_list:
            print(ood_name)

            # Compose max and second max labels from model predictions
            _, testset_output_ind, testset_output_ood = self.load_model_outputs(model_name, ood_name)
            self.output_debug_dict[ood_name] = {'IND': testset_output_ind, 'OOD': testset_output_ood}
            output_ind_pairs = self.compose_pair(testset_output_ind)
            output_ood_pairs = self.compose_pair(testset_output_ood)

            # Compute regret
            trainset, trainset_labels, testset_ind, testset_ood = self.load_model_embedding(model_name, ood_name)
            regret_ind_np, regret_ood_np, trainset_decomp = compute_regret_ind_and_ood(trainset, trainset_labels,
                                                                                       testset_ind, testset_ood,
                                                                                       output_ind_pairs,
                                                                                       output_ood_pairs,
                                                                                       trainset_decomp)

            # Get performance on the dataset
            perf_regret = self.compute_perf_single(model_name, ood_name, regret_ind_np.tolist(), regret_ood_np.tolist())
            perf_df_list.append(perf_regret)

            self.regret_debug_dict[ood_name] = {'IND': regret_ind_np, 'OOD': regret_ood_np}

        df = pd.concat(perf_df_list)
        df.index = pd.RangeIndex(len(df.index))

        debug_dict = {'regret': self.regret_debug_dict, 'output': self.output_debug_dict}
        return df, debug_dict


class BenchmarkPNMLEnsemble(BenchmarkPNML):
    def load_model_embedding(self, model_name: str, testset_name: str) -> (np.ndarray, np.ndarray,
                                                                           np.ndarray, np.ndarray):
        aug_num = 1

        # Trainset load
        trainset_name = model_name.split(' ')[-1].lower()
        trainset_path = osp.join(self.base_dir,
                                 model_name_map[model_name] + '_' + trainset_name + '_' + 'train_pnml.npy')
        trainset = np.load(trainset_path).T

        # Load ensemble results
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

        # Load ensemble results
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

        # Load ensemble results
        testset_ood_list = []
        for i in range(aug_num):
            testset_ood_path = osp.join(self.base_dir,
                                        model_name_map[model_name] + '_' + testset_name_map[
                                            testset_name] + '_' + 'test_pnml_%d.npy' % i)
            testset_ood_list.append(np.load(testset_ood_path).T)
        testset_ood = np.vstack([testset_ood] + testset_ood_list)

        return trainset, trainset_labels, testset_ind, testset_ood


def benchmark_pnml(ood_name_list: list,
                   model_name: str,
                   base_dir_path=osp.join('..', 'output', 'embedding')) -> (pd.DataFrame, dict):
    benchmark_h = BenchmarkPNML(base_dir_path)
    df, debug_dict = benchmark_h.execute(ood_name_list, model_name)
    return df, debug_dict


def cat_benchmark_df(odin_df: pd.DataFrame, lvo_df: pd.DataFrame, pnml_df: pd.DataFrame) -> pd.DataFrame:
    odin_df_dropped = odin_df[odin_df['OOD Datasets'] != 'iSUN']
    odin_df_dropped.index = pd.RangeIndex(len(odin_df_dropped.index))

    pnml_df_dropped = pnml_df[pnml_df['OOD Datasets'] != 'iSUN']
    pnml_df_dropped.index = pd.RangeIndex(len(pnml_df_dropped.index))

    merge_df = pd.concat([odin_df_dropped,
                          lvo_df,
                          pnml_df_dropped
                          ]).sort_index(kind='merge')

    return merge_df.round(1)


if __name__ == '__main__':
    base_dir = osp.join('..', 'output', 'embedding')

    ood_name_benchmark_list = ['Imagenet (crop)',
                               'Imagenet (resize)',
                               'LSUN (crop)',
                               'LSUN (resize)',
                               'iSUN',
                               'Uniform',
                               'Gaussian'
                               ]

    pnml_df_resnet_cifar100, _ = benchmark_pnml(ood_name_benchmark_list, 'DensNet-BC CIFAR100', base_dir)
    print(pnml_df_resnet_cifar100[['AUROC ↑', 'AP-In ↑']])
