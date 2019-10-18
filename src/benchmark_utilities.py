import os.path as osp

import numpy as np
import pandas as pd

from distributions_metrics import calc_performance_in_out_dist
from score_utilities import project_testset, decompose_trainset, load_trainset_features, load_test_products

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
        self.method = 'pNML'
        self.model_suffix = '_1'

    # def load_model_outputs(self, model_name: str, testset_name: str) -> (np.ndarray, np.ndarray, np.ndarray):
    #     trainset_name = model_name.split(' ')[-1].lower()
    #     testset_name_file = testset_name_map[testset_name]
    #     model_file_name = model_name_map[model_name]
    #
    #     # Testset IND
    #     testset_output_ind_path = osp.join(self.base_dir,
    #                                        model_file_name + '_' + trainset_name + '_' + 'test_outputs_pnml{}.npy'.format(
    #                                            self.model_suffix))
    #     print('testset_output_ind_path: ', testset_output_ind_path)
    #     testset_output_ind = np.load(testset_output_ind_path)
    #
    #     # Testset OOD
    #     testset_output_ood_path = osp.join(self.base_dir,
    #                                        model_file_name + '_' + testset_name_file + '_' + 'test_outputs_pnml{}.npy'.format(
    #                                            self.model_suffix))
    #     print('testset_output_ood_path: ', testset_output_ood_path)
    #     testset_output_ood = np.load(testset_output_ood_path)
    #
    #     # Trainset labels load
    #     trainset_labels_path = osp.join(self.base_dir, trainset_name + '_' + 'train_labels.npy')
    #     print('trainset_labels_path: ', trainset_labels_path)
    #     trainset_labels = np.load(trainset_labels_path)
    #
    #     return trainset_labels, testset_output_ind, testset_output_ood

    def load_model_outputs(self, model_name: str, testset_name: str) -> (np.ndarray, np.ndarray, np.ndarray):
        features_path = osp.join('..', 'output', 'features')
        logits_path = osp.join('..', 'output', 'logits')
        testset_name_file = testset_name_map[testset_name]
        model_file_name = model_name_map[model_name]
        model_name, trainset_name = model_file_name.split('_')
        _, testset_ind_pred = load_test_products(model_name, trainset_name, trainset_name, features_path,
                                                 logits_path)
        _, testset_ood_pred = load_test_products(model_name, trainset_name, testset_name_file,
                                                 features_path,
                                                 logits_path)

        trainset_labels_path = osp.join(self.base_dir, trainset_name + '_' + 'train_labels.npy')
        print('trainset_labels_path: ', trainset_labels_path)
        trainset_labels = np.load(trainset_labels_path)
        return trainset_labels, testset_ind_pred, testset_ood_pred

    def load_model_embedding(self, model_name: str, testset_name: str) -> (np.ndarray, np.ndarray,
                                                                           np.ndarray, np.ndarray):
        trainset_name = model_name.split(' ')[-1].lower()
        testset_name_file = testset_name_map[testset_name]
        model_file_name = model_name_map[model_name]

        # todo: keep?
        model_name = model_name_map[model_name].split('_')[0]
        features_path = osp.join('..', 'output', 'features')
        logits_path = osp.join('..', 'output', 'logits')
        trainset, trainset_labels = load_trainset_features(model_name, trainset_name, features_path)
        #
        testset_ind, _ = load_test_products(model_name, trainset_name, trainset_name, features_path,
                                            logits_path)
        testset_ood, _ = load_test_products(model_name, trainset_name, testset_name_file,
                                            features_path,
                                            logits_path)
        # todo: up to here

        # Trainset load
        trainset_path = osp.join(self.base_dir,
                                 model_file_name + '_' + trainset_name + '_' + 'train_pnml{}.npy'.format(
                                     self.model_suffix))
        print('trainset_path: ', trainset_path)
        trainset = np.load(trainset_path).T

        # # Trainset labels load
        # trainset_labels_path = osp.join(self.base_dir, trainset_name + '_' + 'train_labels.npy')
        # trainset_labels = np.load(trainset_labels_path)

        # Testset In-Dist load
        testset_ind_path = osp.join(self.base_dir,
                                    model_file_name + '_' + trainset_name + '_' + 'test_pnml{}.npy'.format(
                                        self.model_suffix))
        print('testset_ind_path: ', testset_ind_path)
        testset_ind = np.load(testset_ind_path).T

        # Testset OOD
        testset_ood_path = osp.join(self.base_dir,
                                    model_file_name + '_' + testset_name_file + '_' + 'test_pnml{}.npy'.format(
                                        ''))#self.model_suffix))
        print('testset_ood_path: ', testset_ood_path)
        testset_ood = np.load(testset_ood_path).T
        return trainset, trainset_labels, testset_ind, testset_ood

    def compute_perf_single(self, model_name: str, ood_name: str, score_ind: list, score_ood: list):
        assert isinstance(score_ind, list) and isinstance(score_ood, list)

        y_score_ind = score_ind + score_ood
        y_score_ind = 1 / np.array(y_score_ind)
        y_true_ind = [True] * len(score_ind) + [False] * len(score_ood)
        perf_regret = calc_performance_in_out_dist(y_true_ind, y_score_ind)

        perf_regret.insert(0, 'Method', [self.method])
        perf_regret.insert(0, 'OOD Datasets', [ood_name])
        perf_regret.insert(0, 'Model', [model_name] * len(perf_regret))

        return perf_regret

    # def compose_pair(self, outputs, first_top: int = 0, second_top: int = 1):
    #     indexes = np.argsort(-outputs, axis=1)
    #     labels_max = indexes[:, first_top]
    #
    #     return labels_max

    def execute(self, ood_name_list: list, model_name: str) -> (pd.DataFrame, dict):
        print('benchmark_pnml. model_name={}'.format(model_name))

        perf_df_list = []

        # Decompose
        trainset, trainset_labels, testset_ind, _ = self.load_model_embedding(model_name, ood_name_list[0])
        trainset_decomp = decompose_trainset(trainset, trainset_labels)

        # Project IND
        _, output_ind_labels, _ = self.load_model_outputs(model_name, ood_name_list[0])
        # output_ind_labels = self.compose_pair(testset_output_ind)
        regret_ind_np = project_testset(testset_ind, output_ind_labels, trainset_decomp)

        for ood_name in ood_name_list:
            print(ood_name)

            # Compose max and second max labels from model predictions
            _, _, output_ood_labels = self.load_model_outputs(model_name, ood_name)
            # output_ood_labels = self.compose_pair(testset_output_ood)

            # Project testset OOD
            _, _, _, testset_ood = self.load_model_embedding(model_name, ood_name)
            regret_ood_np = project_testset(testset_ood, output_ood_labels, trainset_decomp)

            # Get performance on the dataset
            perf_regret = self.compute_perf_single(model_name, ood_name, regret_ind_np.tolist(), regret_ood_np.tolist())
            perf_df_list.append(perf_regret)

        df = pd.concat(perf_df_list)
        df.index = pd.RangeIndex(len(df.index))

        return df


def benchmark_pnml(ood_name_list: list,
                   model_name: str,
                   base_dir_path=osp.join('..', 'output', 'embedding')) -> (pd.DataFrame, dict):
    benchmark_h = BenchmarkPNML(base_dir_path)
    df = benchmark_h.execute(ood_name_list, model_name)
    return df


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
    # base_dir = osp.join('..', 'output', 'embedding_new')
    ood_name_benchmark_list = ['Imagenet (crop)',
                               'Imagenet (resize)',
                               'LSUN (crop)',
                               'LSUN (resize)',
                               'iSUN',
                               'Uniform',
                               'Gaussian'
                               ]

    pnml_df_resnet_cifar100 = benchmark_pnml(ood_name_benchmark_list, 'DensNet-BC CIFAR10', base_dir)
    # print(pnml_df_resnet_cifar100[['AUROC ↑', 'AP-In ↑']])
    print(pnml_df_resnet_cifar100.to_string())
