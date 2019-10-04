import os
import time
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torch.optim as optim
from loguru import logger
from torch import nn
from torch.nn import Parameter
from tqdm import tqdm


def load_my_state_dict(own_state, state_dict):
    for name, param in state_dict.items():
        if name not in own_state:
            continue
        if isinstance(param, Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        own_state[name].copy_(param)
    return own_state


class TrainClass:
    """
    Class which execute train on a DNN model.
    """

    def __init__(self, params_to_train, learning_rate: float,
                 momentum: float, step_size: list, gamma: float, weight_decay: float):
        """
        Initialize train class object.
        :param params_to_train: the parameters of pytorch Module that will be trained.
        :param learning_rate: initial learning rate for the optimizer.
        :param momentum:  initial momentum rate for the optimizer.
        :param step_size: reducing the learning rate by gamma each step_size.
        :param gamma:  reducing the learning rate by gamma each step_size.
        :param weight_decay: L2 regularization.
        """

        self.num_epochs = 20
        self.eval_test_during_train = True
        self.eval_test_in_end = True
        self.print_during_train = True

        # Optimizer
        self.optimizer = optim.SGD(params_to_train,
                                   lr=learning_rate,
                                   momentum=momentum,
                                   weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                        milestones=step_size,
                                                        gamma=gamma)
        self.freeze_batch_norm = True

    def train_model(self, model, dataloaders, num_epochs: int = 10, acc_goal=None):
        """
        Train DNN model using some trainset.
        :param model: the model which will be trained.
        :param dataloaders: contains the trainset for training and testset for evaluation.
        :param num_epochs: number of epochs to train the model.
        :param acc_goal: stop training when getting to this accuracy rate on the trainset.
        :return: trained model (also the training of the models happen inplace)
                 and the loss of the trainset and testset.
        """
        model = model.cuda() if torch.cuda.is_available() else model
        self.num_epochs = num_epochs
        train_loss, train_acc = -1.0, -1.0
        test_loss, test_acc = -1.0, -1.0
        epoch_time = 0
        lr = 0

        # Loop on epochs
        for epoch in range(self.num_epochs):

            epoch_start_time = time.time()
            train_loss, train_acc = self.train(model, dataloaders['train'])
            if self.eval_test_during_train is True:
                test_loss, test_acc = self.test(model, dataloaders['test'])
            epoch_time = time.time() - epoch_start_time

            for param_group in self.optimizer.param_groups:
                lr = param_group['lr']

            logger.info('[%d/%d] [train test] loss =[%f %f], acc=[%f %f], lr=%f, epoch_time=%.2f'
                        % (epoch, self.num_epochs - 1,
                           train_loss, test_loss, train_acc, test_acc,
                           lr, epoch_time))

            # Stop training if desired goal is achieved
            if acc_goal is not None and train_acc >= acc_goal:
                break

        # Print and save
        logger.info('----- [train test] loss =[%f %f], acc=[%f %f] epoch_time=%.2f' %
                    (train_loss, test_loss, train_acc, test_acc, epoch_time))
        return model, train_loss, test_loss

    def train(self, model, train_loader):
        """
        Execute one epoch of training
        :param model: the model that will be trained.
        :param train_loader: contains the trainset to train.
        :return: the loss and accuracy of the trainset.
        """
        model.train()

        # Turn off batch normalization update
        if self.freeze_batch_norm is True:
            model = model.apply(set_bn_eval)

        train_loss = 0
        correct = 0
        # Iterate over dataloaders
        for iter_num, (images, labels) in enumerate(train_loader):
            # Adjust to CUDA
            images = images.cuda() if torch.cuda.is_available() else images
            labels = labels.cuda() if torch.cuda.is_available() else labels

            # Forward
            outputs = model(images)
            loss = self.criterion(outputs, labels)  # Negative log-loss
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            train_loss += loss.item()  # loss sum for all the batch. notice reduction='sum'

            # Back-propagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.scheduler.step()
        train_loss /= len(train_loader.dataset)
        train_acc = correct / len(train_loader.dataset)
        return train_loss, train_acc

    def test(self, model, test_loader):
        """
        Evaluate the performance of the model on the trainset.
        :param model: the model that will be evaluated.
        :param test_loader: testset on which the evaluation will executed.
        :return: the loss and accuracy on the testset.
        """
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, labels in test_loader:
                data = data.cuda() if torch.cuda.is_available() else data
                labels = labels.cuda() if torch.cuda.is_available() else labels

                outputs = model(data)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()  # loss sum for all the batch
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

        test_acc = correct / len(test_loader.dataset)
        test_loss /= len(test_loader.dataset)
        return test_loss, test_acc


def eval_single_sample(model, test_sample_data, temperature: float = 1.0):
    """
    Predict the probabilities assignment of the test sample by the model
    :param model: the model which will evaluate the test sample
    :param test_sample_data: the data of the test sample
    :param temperature: scaling factor of the logits
    :return: prob: probabilities vector. pred: the class prediction of the test sample
    """
    # test_sample = (data, label)

    # Test the sample
    model.eval()
    sample_data = test_sample_data.cuda() if torch.cuda.is_available() else test_sample_data
    output = model(sample_data.unsqueeze(0))

    # Prediction
    pred = output.max(1, keepdim=True)[1]
    pred = pred.item()

    # Extract prob
    prob = F.softmax(output / temperature, dim=-1)
    prob = prob.cpu().detach().numpy().round(16).tolist()[0]
    return prob, pred


def load_pretrained_model(model: nn.Module, pretrained_path: str, dataloaders: dict, is_test=True):
    logger.info('Load pretrained model')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if pretrained_path.endswith('densenet10.pth') or pretrained_path.endswith('densenet100.pth'):
        model_loaded = torch.load(pretrained_path, map_location=device)
        model_loaded = model_loaded['state_dict'] if isinstance(model_loaded, dict) else model_loaded.state_dict()
        state_dict = load_my_state_dict(model.state_dict(), model_loaded)
        model.load_state_dict(state_dict)
    elif pretrained_path == 'wrn28_10_cifar10':
        pass
        # model_loaded = ptcv_get_model("wrn28_10_cifar10", pretrained=True)
        # model.load_state_dict(model_loaded.state_dict())
    elif pretrained_path == 'wrn28_10_cifar100':
        pass
        # model_loaded = ptcv_get_model("wrn28_10_cifar100", pretrained=True)
        # model.load_state_dict(model_loaded.state_dict())
    model.to(device)
    if is_test is True:
        logger.info('Eval test_in_dist dataloader')
        criterion = nn.CrossEntropyLoss(reduction='sum')
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, labels in tqdm(dataloaders['test_in_dist']):
                data, labels = data.to(device), labels.to(device)

                outputs = model(data)
                loss = criterion(outputs, labels)
                test_loss += loss.item()  # loss sum for all the batch
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

        test_acc = correct / len(dataloaders['test_in_dist'].dataset)
        test_loss /= len(dataloaders['test_in_dist'].dataset)
        logger.info(
            'Pretrained model: [Acc Error Loss]=[{:.2f}% {:.2f}% {:.3f}]'.format(100 * test_acc,
                                                                                 100 - 100 * test_acc,
                                                                                 test_loss))
    return model


def execute_basic_training(model_base, dataloaders, params_train, experiment_h):
    if params_train['do_initial_training'] is True:
        logger.info('Execute basic training')
        train_class = TrainClass(filter(lambda p: p.requires_grad, model_base.parameters()), params_train['lr'],
                                 params_train['momentum'], params_train['step_size'], params_train['gamma'],
                                 params_train['weight_decay'])
        train_class.eval_test_during_train = params_train['debug_flags']['eval_test_during_train']
        train_class.freeze_batch_norm = False
        acc_goal = params_train['acc_goal'] if 'acc_goal' in params_train else None
        model_base, train_loss, test_loss = \
            train_class.train_model(model_base, dataloaders, params_train['epochs'], acc_goal)
        torch.save(model_base.state_dict(),
                   os.path.join(logger.output_folder, '%s_model_%f.pt' % (experiment_h.get_exp_name(), train_loss)))
    elif params_train['do_initial_training'] is False:
        logger.info('Load pretrained model')
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model_loaded = torch.load(params_train['pretrained_model_path'], map_location=device)
        model_loaded = model_loaded if isinstance(model_base, OrderedDict) else model_loaded.state_dict()
        state_dict = load_my_state_dict(model_base.state_dict(), model_loaded)
        model_base.load_state_dict(state_dict)
        model_base = model_base.cuda() if torch.cuda.is_available() else model_base

    return model_base


def set_bn_eval(model):
    """
    Freeze batch normalization layers for better control on training
    :param model: the model which the freeze of BN layers will be executed
    :return: None, the freeze is in place on the model.
    """
    classname = model.__class__.__name__
    if classname.find('BatchNorm') != -1:
        model.eval()


def freeze_model_layers(model, max_freeze_layer: int):
    """
    Freeze model layers until max_freeze_layer, all others can be updated
    :param model: to model on which the freeze will be executed
    :param max_freeze_layer: the maximum depth of freeze layer
    :return: model with freeze layers
    """
    for ct, child in enumerate(model.children()):
        if ct <= max_freeze_layer:
            logger.info('Freeze Layer: idx={}, name={}'.format(ct, child))
            for param in child.parameters():
                param.requires_grad = False
            continue
        logger.info('UnFreeze Layer: idx={}, name={}'.format(ct, child))
    return model
