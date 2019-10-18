from collections import OrderedDict

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from torch.nn import functional as F
from torch.utils import data
from torchvision import datasets

from dataset_utilities import transform_cifar_test


class ModelWrapper(pl.LightningModule):

    def __init__(self, model: torch.nn.Module, hparams):
        super().__init__()
        self.model = model
        self.hparams = hparams

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y, reduction='mean')

        _, predicted = torch.max(y_hat.data, 1)
        is_correct = (predicted == y).sum()
        acc = is_correct.double() / len(x)

        loss_svd = 0
        if self.hparams.with_svd:

            for label in torch.unique(y):
                batch_single_class = batch[y == label, :]
                _, s, _ = torch.svd(batch_single_class, compute_uv=False)
                eta = s ** 2
                loss_svd += 1 / len(batch_single_class) * (eta[1:]/eta[0]).sum()

            loss_svd /= len(label)

            loss += loss_svd
        lr = 0
        for param_group in self.trainer.optimizers[0].param_groups:
            lr = param_group['lr']

        return {'loss': loss,
                'log': {'phase': 'Train',
                        'loss': loss,
                        'loss_svd': loss_svd,
                        'acc': acc,
                        'lr': lr,
                        'batch_nb': batch_nb,
                        'nb_training_batches': self.trainer.nb_training_batches,
                        'nb_epochs': self.trainer.max_nb_epochs}}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)

        _, predicted = torch.max(y_hat.data, 1)
        is_correct = (predicted == y).sum()
        acc = is_correct.double() / len(x)
        return {'val_loss': F.cross_entropy(y_hat, y, reduction='mean'),
                'val_acc': acc}

    def validation_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs
        :param outputs: list of individual outputs of each validation step
        :return:
        """
        val_loss_mean = 0
        val_acc_mean = 0
        for output in outputs:
            val_loss_mean += output['val_loss']
            val_acc_mean += output['val_acc']

        val_loss_mean /= len(outputs)
        val_acc_mean /= len(outputs)

        # Return for the collation function validation_end
        # everything must be a tensor
        output = OrderedDict({
            'log': {'phase': 'Val',
                    'loss': val_loss_mean,
                    'acc': val_acc_mean,
                    'nb_epochs': self.trainer.max_nb_epochs},
            'val_loss': val_loss_mean,
            'val_acc': val_acc_mean,
        })

        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.hparams.lr,
                                     weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.hparams.step_size)
        return [optimizer], [scheduler]

    @pl.data_loader
    def train_dataloader(self):
        if self.hparams.dataset == 'cifar10':
            trainset = datasets.CIFAR10(root=self.hparams.data_dir,
                                        train=True,
                                        download=True,
                                        transform=transform_cifar_test)  # transform_cifar_train)
        else:
            trainset = datasets.CIFAR100(root=self.hparams.data_dir,
                                         train=True,
                                         download=True,
                                         transform=transform_cifar_test)  # transform_cifar_train)
        trainloader = data.DataLoader(trainset,
                                      batch_size=self.hparams.batch_size,
                                      shuffle=True,
                                      num_workers=self.hparams.num_workers)

        return trainloader

    @pl.data_loader
    def val_dataloader(self):
        if self.hparams.dataset == 'cifar10':
            testset = datasets.CIFAR10(root=self.hparams.data_dir,
                                       train=False,
                                       download=True,
                                       transform=transform_cifar_test)
        else:
            testset = datasets.CIFAR100(root=self.hparams.data_dir,
                                        train=False,
                                        download=True,
                                        transform=transform_cifar_test)
        testloader = data.DataLoader(testset,
                                     batch_size=self.hparams.batch_size,
                                     shuffle=False,
                                     num_workers=self.hparams.num_workers)
        return testloader

    def load_from_metrics(cls, weights_path, tags_csv):
        pass


class MyTrainer(Trainer):

    def restore_state_if_checkpoint_exists(self, model):
        pass
