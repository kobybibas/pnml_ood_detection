import argparse
import json
import os

from loguru import logger

from experimnet_utilities import Experiment
from experimnet_utilities import experiment_name_valid
from odin_utilities import execute_odin_baseline, perturbate_dataset
from pnml_utilities import extract_features
from result_tracker_utilities import ResultTracker
from train_utilities import test_pretrained_model

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
    # Load pretrained model
    model = experiment_h.get_model()
    model = test_pretrained_model(model, dataloaders, is_test=params['eval_pretrained'])
    logger.info(model)

    if params['odin']['is_odin'] is True:
        logger.info('Execute dataset perturbation')
        dataloaders['test'] = perturbate_dataset(model, dataloaders['test'],
                                                 params['odin']['magnitude'], params['odin']['temperature'])
    if params['pnml']['is_pnml'] is True:
        logger.info('Execute pNML')
        extract_features(model, dataloaders, experiment_h, tracker)

    if params['odin']['is_odin'] is True:
        logger.info('Execute ODIN')
        execute_odin_baseline(model, dataloaders['test'], params['odin']['temperature'], experiment_h, tracker)

    logger.info('Finished!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Applications of Deep PNML')
    parser.add_argument('-t', '--experiment_type', default='densenet_cifar10',
                        choices=experiment_name_valid,
                        help='Type of experiment to execute',
                        type=str)
    parser.add_argument('-testset',
                        help='Testset to evaluate',
                        type=str)
    parser.add_argument('-prefix',
                        default='',
                        help='Output directory name prefix',
                        type=str)
    parser.add_argument('-num_workers',
                        help='CPU workers for dataloader',
                        type=int)
    parser.add_argument('-temperature',
                        help='Scale of logits',
                        type=float)
    parser.add_argument('-is_odin',
                        help='whether to execute ODIN baseline',
                        type=lambda s: s.lower() in ['true', 't', 'yes', '1'])
    parser.add_argument('-is_pnml',
                        help='whether to execute pNML scheme',
                        type=lambda s: s.lower() in ['true', 't', 'yes', '1'])

    args = vars(parser.parse_args())

    # Available experiment_type
    # 'densenet_cifar10'
    # 'densenet_cifar100'
    # 'resnet_cifar10'
    # 'resnet_cifar100'

    run_experiment(args)
    print('Finish experiment')

    # To run odin baselines:
    # python main.py -testset isun -is_odin True -is_pnml False prefix baseline_odin_
