import argparse
import json
import os

import numpy as np
from loguru import logger

from dataset_utilities import extract_features
from experimnet_utilities import Experiment
from pnml_utilities import execute_pnml_on_testset
from result_tracker_utilities import ResultTracker
from train_utilities import execute_basic_training

"""
Example of running:
CUDA_VISIBLE_DEVICES=0 python main.py -t densenet_cifar10
"""


def update_dict_recursive(params_org, key, value):
    for k, v in zip(params_org.keys(), params_org.values()):
        if isinstance(v, dict):
            update_dict_recursive(v, key, value)
        elif k == key:
            logger.info('Change param: {}={}'.format(key, value))
            params_org[k] = value


def run_experiment(args: dict):
    ################
    # Load training params
    with open(os.path.join('params.json')) as f:
        params = json.load(f)

    ################
    # Class that depends ins the experiment type
    experiment_h = Experiment(args['experiment_type'], params)
    params = experiment_h.get_params()
    for key, value in args.items():
        if value is None:
            continue
        update_dict_recursive(params, key, value)

    ################
    # Create logger and save params to output folder
    tracker = ResultTracker(experiment_name=os.path.join(args['prefix'] + experiment_h.get_exp_name()),
                            output_root=os.path.join('..', 'output'))
    logger.add(os.path.join(tracker.output_folder, 'log_{}_{}.log'.format(args['experiment_type'],
                                                                          tracker.unique_time)))
    logger.info('OutputDirectory: {}'.format(tracker.output_folder))
    with open(os.path.join(tracker.output_folder, 'params.json'), 'w', encoding='utf8') as outfile:
        outfile.write(json.dumps(params, indent=4, sort_keys=True))
    logger.info(json.dumps(params, indent=4, sort_keys=True))

    ################
    # Load datasets
    data_folder = os.path.join('..', 'data')
    logger.info('Load datasets: %s' % data_folder)
    dataloaders = experiment_h.get_dataloaders(data_folder)

    ################
    # Run basic training- so the base model will be in the same conditions as NML model
    model_base = experiment_h.get_model()
    params_init_training = params['initial_training']
    params_init_training['debug_flags'] = params['debug_flags']
    model_erm = execute_basic_training(model_base, dataloaders, params_init_training, experiment_h)

    # ################
    # # Freeze layers
    # logger.info('Freeze layer: %d' % params['freeze_layer'])
    # model_erm = freeze_model_layers(model_erm, params['freeze_layer'])
    params_preprocess = params['preprocess']
    np.save('../output/cifar100_train_labels.npy', dataloaders['train'].dataset.targets)
    if params['is_split_model'] is True:
        dataloaders['train'].dataset, dataloaders['test'].dataset = extract_features(model_erm, dataloaders,
                                                                                     params_preprocess['is_preprocess'],
                                                                                     params_preprocess[
                                                                                         'temperature_preprocess'],
                                                                                     params_preprocess['magnitude'])
        for set_type, dataloader in dataloaders.items():
            set_name = experiment_h.trainset_name if set_type == 'train' else experiment_h.testset_name
            file_name = os.path.join('..', 'output',
                                     '{}_{}_{}{}.npy'.format(args['experiment_type'], set_name, set_type,
                                                             '_odin' if params_preprocess['is_preprocess'] else ''))
            logger.info('Saving to {}'.format(file_name))
            np.save(file_name, dataloader.dataset.data.numpy())
        np.save(os.path.join('..', 'output', '{}_train_labels.npy'.format(experiment_h.trainset_name)),
                dataloaders['train'].dataset.targets)

    return

    ############################
    # Iterate over test dataset
    logger.info('Execute pNML')
    params_fit_to_sample = params['fit_to_sample']
    params_fit_to_sample['debug_flags'] = params['debug_flags']
    execute_pnml_on_testset(model_erm, experiment_h, params_fit_to_sample, params_preprocess, dataloaders, tracker)
    logger.info('Finish All!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Applications of Deep PNML')
    parser.add_argument('-t', '--experiment_type', default='densenet_cifar10',
                        help='Type of experiment to execute',
                        type=str)
    parser.add_argument('-testset',
                        help='Testset to evaluate',
                        type=str)
    parser.add_argument('-epochs',
                        help='Epochs for fine-tuning',
                        type=int)
    parser.add_argument('-lr',
                        help='Learning rate for fine-tuning',
                        type=int)
    parser.add_argument('-prefix',
                        default='',
                        help='output directory name prefix',
                        type=str)
    parser.add_argument('-num_workers',
                        help='CPU workers',
                        type=int)
    parser.add_argument('-temperature',
                        help='Scale the logits',
                        type=float)
    parser.add_argument('-is_preprocess',
                        help='whether to preprocess',
                        type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
    parser.add_argument('-test_start_idx',
                        help='First test index to evaluate',
                        type=int)
    parser.add_argument('-test_end_idx',
                        help='Final test index to evaluate',
                        type=int)

    args = vars(parser.parse_args())

    # Available experiment_type:
    #   'pnml_cifar10'
    #   'random_labels'
    #   'out_of_dist_svhn'
    #   'out_of_dist_noise'
    #   'pnml_mnist'
    #   'pnml_cifar10_lenet'

    # 'densenet_cifar10'

    run_experiment(args)
    print('Finish experiment')

    # To run odin baselines:
    # python main.py -testset isun -is_preprocess True -epochs 0 -prefix baseline_odin_
    # python main.py -testset cifar10 -is_preprocess True  -epochs 0 -prefix baseline_odin_
