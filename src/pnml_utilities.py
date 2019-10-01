import time
from copy import deepcopy

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader

from dataset_utilities import insert_sample_to_dataset
from experimnet_utilities import Experiment
from result_tracker_utilities import ResultTracker
from train_utilities import TrainClass
from train_utilities import eval_single_sample


def execute_pnml_on_testset(model_base, experiment_h: Experiment, params_fit_to_sample: dict, params_preprocess: dict,
                            dataloaders: dict,
                            tracker: ResultTracker):
    idx_start, idx_end = params_fit_to_sample['test_start_idx'], params_fit_to_sample['test_end_idx']
    idx_end = min(idx_end + 1, len(dataloaders['test'].dataset.targets))
    testloader_single = deepcopy(dataloaders['test'])
    assert isinstance(testloader_single, DataLoader)
    for idx in range(idx_start, idx_end):
        time_start_idx = time.time()

        # Extract a sample from test dataset and check output of base model
        test_data = dataloaders['test'].dataset.data[idx]
        test_label = dataloaders['test'].dataset.targets[idx]
        test_data_transformed = dataloaders['test'].dataset.transform(test_data)

        # Evaluate with base model
        temperature = params_preprocess["temperature_preprocess"] if params_preprocess['is_preprocess'] else 1.0
        prob_org, _ = eval_single_sample(model_base, test_data_transformed, temperature)
        tracker.add_org_prob_to_results_dict(idx, prob_org, test_label)

        # NML training- train the model with test sample
        if params_fit_to_sample['epochs'] > 0:
            execute_pnml_training(params_fit_to_sample, dataloaders,
                                  test_data, test_label, idx,
                                  model_base, tracker)
        # Log and save
        tracker.save_json_file()
        time_idx = time.time() - time_start_idx
        logger.info('----- Finish {} idx = {}/{}, time={:.2f}[sec] ----'.format(experiment_h.get_exp_name(),
                                                                                idx, idx_end - 1,
                                                                                time_idx))


def execute_pnml_training(train_params: dict, dataloaders_input: dict,
                          sample_test_data, sample_test_true_label, idx: int,
                          model_base_input, tracker: ResultTracker):
    """
    Execute the PNML procedure: for each label train the model and save the prediction afterword.
    :param train_params: parameters of training the model for each label
    :param dataloaders_input: dataloaders which contains the trainset
    :param sample_test_data: the data of the test sample that will be evaluated
    :param sample_test_true_label: the true label of the test sample
    :param idx: the index in the testset dataset of the test sample
    :param model_base_input: the base model from which the train will start
    :param tracker: logger class to print logs and save results to file
    :return: None
    """

    # Check train_params contains all required keys
    required_keys = ['lr', 'momentum', 'step_size', 'gamma', 'weight_decay', 'epochs', 'temperature']
    for key in required_keys:
        if key not in train_params:
            logger.error('The key: %s is not in train_params' % key)
            raise ValueError('The key: %s is not in train_params' % key)

    # Iteration of all labels
    labels_to_train = np.unique(dataloaders_input['train'].dataset.targets).tolist()
    trainset_empty = deepcopy(dataloaders_input['train'].dataset)
    data_size = trainset_empty.data.shape
    trainset_empty.data = np.empty(data_size) if isinstance(trainset_empty.data,
                                                            np.ndarray) else trainset_empty.data.new_empty
    assert isinstance(labels_to_train, list)
    for trained_label in labels_to_train:
        time_trained_label_start = time.time()

        # Insert test sample to train dataset
        trainloader_with_sample = insert_sample_to_dataset(dataloaders_input['train'],
                                                           trainset_empty,
                                                           sample_test_data, trained_label,
                                                           train_params['batches_num'])
        dataloaders = {'train': trainloader_with_sample, 'test': dataloaders_input['test']}

        # Train model
        model = deepcopy(model_base_input)
        model.fc1 = torch.nn.Linear(model.fc.in_features, model.fc.in_features)  # initialize last layer
        model.fc2 = torch.nn.Linear(model.fc.in_features, model.fc.out_features)  # initialize last layer
        train_class = TrainClass(filter(lambda p: p.requires_grad, model.parameters()), train_params['lr'],
                                 train_params['momentum'], train_params['step_size'], train_params['gamma'],
                                 train_params['weight_decay'])
        train_class.eval_test_during_train = train_params['debug_flags']['eval_test_during_train']
        train_class.freeze_batch_norm = True
        model, train_loss, test_loss = train_class.train_model(model, dataloaders, train_params['epochs'])

        # Execute transformation
        temperature = train_params["temperature"]
        sample_test_data_trans = dataloaders['test'].dataset[idx][0]
        prob, pred = eval_single_sample(model, sample_test_data_trans, temperature)

        # Save to file
        tracker.add_entry_to_results_dict(idx, str(trained_label), prob, train_loss, test_loss)
        logger.info('idx=%d label [trained true]=[%d %d] predict=[%d], loss [train, test]=[%f %f], time=%4.2f[s]'
                    % (idx, trained_label, sample_test_true_label,
                       int(np.argmax(prob)),
                       train_loss, test_loss,
                       time.time() - time_trained_label_start))

    # regret
    prob = [tracker.results_dict[str(idx)][str(cls)]['prob'][cls] for cls in labels_to_train]
    regret = np.log10(np.sum(prob))
    logger.info('idx = {}, Regret = {:.5f}, Genies prob {}'.format(idx, regret, np.round(prob, 3)))
