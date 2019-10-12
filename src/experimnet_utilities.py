import os
import types

from loguru import logger
from pytorchcv.model_provider import get_model as ptcv_get_model
import os.path as osp
from dataset_utilities import create_cifar10_dataloaders, create_cifar100_dataloaders
from dataset_utilities import create_image_folder_trainloader
from dataset_utilities import create_uniform_noise_dataloaders, create_gaussian_noise_dataloaders
from glob import glob
from model_utilities import add_feature_extractor_method

experiment_name_valid = [
    'densenet_cifar10',
    'densenet_cifar100',
    'resnet_cifar10',
    'resnet_cifar100'
]

testsets_name = [
    'cifar10',
    'cifar100',
    # Tiny - ImageNet(crop)
    'Imagenet',
    # Tiny-ImageNet (resize)
    'Imagenet_resize',
    # LSUN (crop)
    'LSUN',
    # LSUN (resize)
    'LSUN_resize',
    # iSUN
    'iSUN',
    # Gaussian noise
    'Gaussian',
    # Uniform noise
    'Uniform'
]


class Experiment:
    def __init__(self, exp_type: str, params: dict):
        if exp_type not in experiment_name_valid:
            logger.error('Experiment name {} is not valid'.format(exp_type))
            raise NameError('No experiment type: %s' % type)
        self.params = params
        self.exp_type = exp_type
        self.executed_get_params = False
        self.trainset_name = ''
        self.testset_name = ''

    def get_params(self):
        self.params = self.params[self.exp_type]
        self.executed_get_params = True
        return self.params

    def get_dataloaders(self, data_folder: str = os.path.join('..', 'data')):
        if self.executed_get_params is False:
            _ = self.get_params()

        # Load trainset
        trainloader, testloader, testloader_in_distribution = None, None, None
        if self.exp_type.endswith('cifar10'):
            self.trainset_name = 'cifar10'
            trainloader, testloader_cifar10, _ = create_cifar10_dataloaders(data_folder,
                                                                            self.params['batch_size'],
                                                                            self.params['num_workers'])
            testloader_in_distribution = testloader_cifar10
        elif self.exp_type.endswith('cifar100'):
            self.trainset_name = 'cifar100'
            trainloader, testloader_cifar100, _ = create_cifar100_dataloaders(data_folder,
                                                                              self.params['batch_size'],
                                                                              self.params['num_workers'])
            testloader_in_distribution = testloader_cifar100
        else:
            ValueError('Trainset is not available')

        # Load testset
        if self.params['testset'] in ['cifar10']:
            self.testset_name = 'cifar10'
            testloader = testloader_cifar10
        elif self.params['testset'] in ['cifar100']:
            self.testset_name = 'cifar100'
            testloader = testloader_cifar100
        elif self.params['testset'] in ['iSUN']:
            self.testset_name = 'iSUN'
            testloader = create_image_folder_trainloader(os.path.join('..', 'data', 'iSUN'),
                                                         self.params['batch_size'],
                                                         self.params['num_workers'],
                                                         self.trainset_name)
        elif self.params['testset'] in ['Imagenet']:
            self.testset_name = 'Imagenet'
            testloader = create_image_folder_trainloader(os.path.join('..', 'data', 'Imagenet'),
                                                         self.params['batch_size'],
                                                         self.params['num_workers'],
                                                         self.trainset_name
                                                         )
        elif self.params['testset'] in ['Imagenet_resize']:
            self.testset_name = 'Imagenet_resize'
            testloader = create_image_folder_trainloader(os.path.join('..', 'data', 'Imagenet_resize'),
                                                         self.params['batch_size'],
                                                         self.params['num_workers'],
                                                         self.trainset_name)
        elif self.params['testset'] in ['LSUN']:
            self.testset_name = 'LSUN'
            testloader = create_image_folder_trainloader(os.path.join('..', 'data', 'LSUN'),
                                                         self.params['batch_size'],
                                                         self.params['num_workers'],
                                                         self.trainset_name)
        elif self.params['testset'] in ['LSUN_resize']:
            self.testset_name = 'LSUN_resize'
            testloader = create_image_folder_trainloader(os.path.join('..', 'data', 'LSUN_resize'),
                                                         self.params['batch_size'],
                                                         self.params['num_workers'],
                                                         self.trainset_name)
        elif self.params['testset'] in ['Uniform']:
            self.testset_name = 'Uniform'
            testloader = create_uniform_noise_dataloaders(self.params['batch_size'],
                                                          self.params['num_workers'],
                                                          self.trainset_name)
        elif self.params['testset'] in ['Gaussian']:
            self.testset_name = 'Gaussian'
            testloader = create_gaussian_noise_dataloaders(self.params['batch_size'],
                                                           self.params['num_workers'],
                                                           self.trainset_name)
        else:
            ValueError('{} testset is not available'.format(self.params['testset']))
        assert trainloader is not None
        assert testloader is not None

        # Assign transforms

        dataloaders = {'train': trainloader, 'test': testloader, 'test_in_dist': testloader_in_distribution}
        return dataloaders

    def get_model(self):
        model_name, dataset_name = self.exp_type.split('_')

        # Get list of pretrained models
        model_list = glob(osp.join(self.params['pretrained_path'], '{}_{}_*.pth'.format(model_name, dataset_name)))
        model_list.sort()

        if model_name == 'densenet':
            model = ptcv_get_model("densenet100_k12_bc_%s" % dataset_name, pretrained=True)
        elif model_name == 'resnet':
            model = ptcv_get_model("wrn28_10_%s" % dataset_name, pretrained=True)
        else:
            raise NameError('No model for experiment type: %s' % self.exp_type)

        model = add_feature_extractor_method(model)
        model.eval()
        return model, model_list

    def get_exp_name(self):
        return self.exp_type + '_' + self.params['testset']
