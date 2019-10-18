import time
from pytorch_lightning.logging import LightningLoggerBase, rank_zero_only


class MyLogger(LightningLoggerBase):

    def __init__(self, logger):
        self.logger = logger
        self.time_start = time.time()

    def info(self, *args, **kwargs):
        self.logger.info(*args, **kwargs)

    def debug(self, *args, **kwargs):
        self.logger.debug(*args, **kwargs)

    def error(self, *args, **kwargs):
        self.logger.error(*args, **kwargs)

    def add(self, *args, **kwargs):
        self.logger.add(*args, **kwargs)

    @rank_zero_only
    def log_hyperparams(self, params):
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step_num):
        if step_num == 0:
            self.time_start = time.time()

        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        if metrics['phase'] == 'Train':
            self.logger.info('    {}: [{:03d}/{:03d}] [Loss Acc]=[{:.3f} {:.3f}] lr={}'.format(
                metrics['phase'], metrics['batch_nb'], metrics['nb_training_batches'], metrics['loss'], metrics['acc'],
                metrics['lr']))
            self.logger.info('Loss svd: ', metrics['loss_svd'])

        elif metrics['phase'] == 'Val':
            self.logger.info("Epoch [{:02d}/{:02d}]: {}. [Loss Acc]=[{:.3f} {:.3f}] {:.2f} sec".format(
                metrics['epoch'], metrics['nb_epochs'] - 1, metrics['phase'],
                metrics['loss'], metrics['acc'], time.time() - self.time_start))
            self.time_start = time.time()
