import itertools
import logging
import os
import os.path as osp

import hydra
import pytorch_lightning as pl
import torch
import yaml
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from dataset_utils import get_dataloaders
from lit_utils.odin_lit_utils import LitOdin

logger = logging.getLogger(__name__)


def optimize_odin_to_dataset(lit_model_h: LitOdin, trainer, ind_loader: DataLoader, ood_loader: DataLoader,
                             ood_name: str,
                             epsilons: list, temperatures: list, is_dev_run: bool = False) -> (dict, dict):
    best_odin_tnr, best_odin_eps, best_odin_temperature = 0.0, 0.0, 1.0
    best_pnml_tnr, best_pnml_eps, best_pnml_temperature = 0.0, 0.0, 1.0
    for i, (temperature, eps) in enumerate(itertools.product(temperatures, epsilons)):

        # Eval ind score
        lit_model_h.set_ood(lit_model_h.ind_name)
        trainer.test(test_dataloaders=ind_loader, ckpt_path=None, verbose=False)

        # Eval ood score
        lit_model_h.set_ood(ood_name)
        trainer.test(test_dataloaders=ood_loader, ckpt_path=None, verbose=False)

        # Get results
        baseline_res, pnml_res = lit_model_h.get_performance()
        odin_tnr = float(baseline_res['TNR at TPR 95%'])
        pnml_tnr = float(pnml_res['TNR at TPR 95%'])

        logger.info('[eps temperature odin_tnr pnml_tnr]=[{} {} {:.3f} {:.3f}]'.format(eps, temperature,
                                                                                       odin_tnr, pnml_tnr))
        if odin_tnr > best_odin_tnr:
            logger.info('    New best odin_tnr.')
            best_odin_eps = eps
            best_odin_temperature = temperature
            best_odin_tnr = odin_tnr
        if pnml_tnr > best_pnml_tnr:
            logger.info('    New best pnml_tnr.')
            best_pnml_eps = eps
            best_pnml_temperature = temperature
            best_pnml_tnr = odin_tnr

        if is_dev_run:
            break

    odin_dict = {'epsilon': best_odin_eps, 'temperature': best_odin_temperature, 'tnr': round(best_odin_tnr, 2)}
    pnml_dict = {'epsilon': best_pnml_eps, 'temperature': best_pnml_temperature, 'tnr': round(best_pnml_tnr, 2)}
    return odin_dict, pnml_dict


@hydra.main(config_path="../configs", config_name="optimize_odin")
def optimize_odin(cfg: DictConfig):
    logger.info(cfg)
    out_dir = os.getcwd()
    os.chdir(hydra.utils.get_original_cwd())
    logger.info('torch.cuda.is_available={}'.format(torch.cuda.is_available()))

    # Load datasets
    logger.info('Load datasets: {}'.format(cfg.data_dir))
    loaders_dict = get_dataloaders(cfg.model, cfg.trainset, cfg.data_dir, cfg.batch_size,
                                   cfg.num_workers if cfg.dev_run is False else 0)
    num_loaders = len(loaders_dict)

    # Get ind loaders
    train_loader, ind_loader = loaders_dict.pop('trainset'), loaders_dict.pop(cfg.trainset)

    # Initialize trainer
    trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else 0,
                         fast_dev_run=cfg.dev_run,
                         num_sanity_val_steps=0,
                         checkpoint_callback=False,
                         max_epochs=1,
                         default_root_dir=out_dir,
                         limit_test_batches=int(round(cfg.num_samples / cfg.batch_size)  # use only validation samples
                                                ))
    lit_model_h = LitOdin(cfg.model, cfg.trainset, out_dir)
    trainer.fit(lit_model_h, train_dataloader=train_loader)

    # Optimize odin_pnml for each dataset
    odin_dicts, pnml_dicts = {}, {}
    for i, (ood_name, ood_loader) in enumerate(loaders_dict.items()):
        odin_dict, pnml_dict = optimize_odin_to_dataset(lit_model_h, trainer, ind_loader, ood_loader,
                                                        ood_name, cfg.epsilons, cfg.temperatures,
                                                        is_dev_run=cfg.dev_run)
        logger.info('[{}/{}] {}: odin={} pnml={}'.format(i, num_loaders - 1, ood_name, odin_dict, pnml_dict))

        odin_dicts[ood_name] = odin_dict
        pnml_dicts[ood_name] = pnml_dict
        if cfg.dev_run:
            break

    # Save results to file
    for method_name, method_dict in [('odin_vanilla', odin_dicts), ('odin_pnml', pnml_dicts)]:
        method_out_dir = osp.join(out_dir, method_name)
        os.makedirs(method_out_dir, exist_ok=True)
        with open(osp.join(method_out_dir, f'{cfg.model}_{cfg.trainset}.yaml'), 'w') as f:
            yaml.dump(method_dict, f, sort_keys=True)

    logger.info('Finished! out_dir={}'.format(out_dir))


if __name__ == "__main__":
    optimize_odin()
