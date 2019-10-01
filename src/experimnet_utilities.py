import os

from dataset_utilities import create_cifar10_dataloaders, create_cifar100_dataloaders
from dataset_utilities import create_image_folder_trainloader
from dataset_utilities import dataloaders_noise
from densnet_splitted import DensNetSplit
from mpl import Net
from resnet import resnet20, load_pretrained_resnet20_cifar10_model
from wide_resnet import WideResNet
from densenet import DenseNet3

testsets_name = [
    # Tiny - ImageNet(crop)
    'Imagenet'
    # Tiny-ImageNet (resize)
    'Imagenet_resize'
    # LSUN (crop)
    'LSUN'
    # LSUN (resize)
    'LSUN_resize'
    # iSUN
    'iSUN'
]


class Experiment:
    def __init__(self, exp_type: str, params: dict):
        if exp_type not in ['densenet_cifar10',
                            'densenet_cifar100'
                            ]:
            raise NameError('No experiment type: %s' % type)
        self.params = params
        self.exp_type = exp_type
        self.executed_get_params = False
        self.trainset_name = ''
        self.testset_name = ''


    def get_params(self):
        debug_flags = self.params['debug_flags']
        self.params = self.params[self.exp_type]
        self.params['debug_flags'] = debug_flags
        self.executed_get_params = True
        return self.params

    def get_dataloaders(self, data_folder: str = os.path.join('..', 'data')):
        if self.executed_get_params is False:
            _ = self.get_params()

        # Load trainset
        trainloader, testloader = None, None
        if self.exp_type.endswith('cifar10'):
            self.trainset_name = 'cifar10'
            trainloader, testloader_cifar10, _ = create_cifar10_dataloaders(data_folder,
                                                                            self.params['batch_size'],
                                                                            self.params['num_workers'])
        elif self.exp_type.endswith('cifar100'):
            self.trainset_name = 'cifar100'
            trainloader, testloader_cifar100, _ = create_cifar100_dataloaders(data_folder,
                                                                              self.params['batch_size'],
                                                                              self.params['num_workers'])
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
                                                         self.params['num_workers'])
        elif self.params['testset'] in ['Imagenet']:
            self.testset_name = 'Imagenet'
            testloader = create_image_folder_trainloader(os.path.join('..', 'data', 'Imagenet'),
                                                         self.params['batch_size'],
                                                         self.params['num_workers'])
        elif self.params['testset'] in ['Imagenet_resize']:
            self.testset_name = 'Imagenet_resize'
            testloader = create_image_folder_trainloader(os.path.join('..', 'data', 'Imagenet_resize'),
                                                         self.params['batch_size'],
                                                         self.params['num_workers'])
        elif self.params['testset'] in ['LSUN']:
            self.testset_name = 'LSUN'
            testloader = create_image_folder_trainloader(os.path.join('..', 'data', 'LSUN'),
                                                         self.params['batch_size'],
                                                         self.params['num_workers'])
        elif self.params['testset'] in ['LSUN_resize']:
            self.testset_name = 'LSUN_resize'
            testloader = create_image_folder_trainloader(os.path.join('..', 'data', 'LSUN_resize'),
                                                         self.params['batch_size'],
                                                         self.params['num_workers'])
        elif self.params['testset'] in ['noise']:
            self.testset_name = 'noise'
            testloader = dataloaders_noise(data_folder,
                                           self.params['batch_size'],
                                           self.params['num_workers'])
        else:
            ValueError('{} testset is not available'.format(self.params['testset']))
        assert trainloader is not None
        assert testloader is not None

        dataloaders = {'train': trainloader, 'test': testloader}
        return dataloaders

    def get_model(self):

        if self.exp_type.startswith('densenet'):
            num_classes = 10 if self.exp_type.endswith('10') else 100
            if self.params['is_split_model'] is True:

                model = DensNetSplit(depth=100, growth_rate=12, num_classes=num_classes)
                model.set_feature_extractor_layer(self.params['feature_extractor_layer'])
            else:
                model = DenseNet3(depth=100, growth_rate=12, num_classes=num_classes)

        elif self.exp_type == 'pnml_cifar10':
            model = load_pretrained_resnet20_cifar10_model(resnet20())
        elif self.exp_type == 'random_labels':
            model = WideResNet()
        elif self.exp_type == 'out_of_dist_svhn':
            model = load_pretrained_resnet20_cifar10_model(resnet20())
        elif self.exp_type == 'out_of_dist_noise':
            model = load_pretrained_resnet20_cifar10_model(resnet20())
        elif self.exp_type == 'pnml_mnist':
            model = Net()
        elif self.exp_type == 'adversarial':
            model = load_pretrained_resnet20_cifar10_model(resnet20())
        else:
            raise NameError('No experiment type: %s' % self.exp_type)

        return model

    def get_exp_name(self):
        return self.exp_type + '_' + self.params['testset']
