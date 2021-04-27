import logging
import os
import os.path as osp
import time

import hydra
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from omegaconf import DictConfig

from dataset_utils import get_dataloaders
from lit_utils.baseline_lit_utils import LitBaseline
from lit_utils.energy_lit_utils import LitEnergy
from lit_utils.gram_lit_utils import LitGram
from lit_utils.odin_lit_utils import LitOdin
from lit_utils.oecc_lit_utils import LitOecc
from method_utils import execute_baseline, execute_odin

logger = logging.getLogger(__name__)


def merge_results(baseline_df, pnml_df, cfg):
    merged = pd.merge(baseline_df, pnml_df, how='inner', on='ood_name', suffixes=[f'_{cfg.method}', '_pnml'])
    merged.insert(0, 'ood_name', merged.pop('ood_name'))
    merged.insert(0, 'ind_name', cfg.trainset)
    merged.insert(0, 'model', cfg.model)
    merged = merged.round(2)
    return merged


def load_odin_parmas(cfg):
    # Load odin params just in case
    path = osp.join('..', 'configs', cfg.odin_vanilla_path)
    logger.info(f'Load {path}')
    with open(path, 'r')as stream:
        odin_vanilla = yaml.safe_load(stream)
    path = osp.join('..', 'configs', cfg.odin_pnml_path)
    logger.info(f'Load {path}')
    with open(path, 'r')as stream:
        odin_pnml = yaml.safe_load(stream)
    return odin_vanilla, odin_pnml


@hydra.main(config_path="../configs", config_name="execute_method")
def run_experiment(cfg: DictConfig):
    t0 = time.time()
    logger.info(cfg)
    out_dir = os.getcwd()
    os.chdir(hydra.utils.get_original_cwd())
    logger.info(f'out_dir={out_dir}')
    pl.seed_everything(cfg.seed)

    # Load datasets
    t1 = time.time()
    loaders_dict = get_dataloaders(cfg.model,
                                   cfg.trainset, cfg.data_dir, cfg.batch_size,
                                   cfg.num_workers if cfg.dev_run is False else 0,
                                   cfg.dev_run)
    logger.info('Finish load datasets in {:.2f} sec'.format(time.time() - t1))

    # Initialize trainer
    trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else 0,
                         fast_dev_run=cfg.dev_run,
                         num_sanity_val_steps=0,
                         checkpoint_callback=False,
                         max_epochs=1, amp_level='O2',
                         default_root_dir=out_dir)

    # Execute method
    if cfg.method == 'baseline':
        lit_model_h = LitBaseline(cfg.model, cfg.trainset, out_dir)
        baseline_df, pnml_df = execute_baseline(cfg, lit_model_h, trainer, loaders_dict)
    elif cfg.method == 'odin':
        lit_model_h = LitOdin(cfg.model, cfg.trainset, out_dir)
        baseline_df, pnml_df = execute_odin(cfg, lit_model_h, trainer, loaders_dict)
    elif cfg.method == 'gram':
        lit_model_h = LitGram(cfg.model, cfg.trainset, out_dir)
        baseline_df, pnml_df = execute_baseline(cfg, lit_model_h, trainer, loaders_dict)
    elif cfg.method == 'energy':
        lit_model_h = LitEnergy(cfg.model, cfg.trainset, out_dir)
        baseline_df, pnml_df = execute_baseline(cfg, lit_model_h, trainer, loaders_dict)
    elif cfg.method == 'oecc':
        lit_model_h = LitOecc(cfg.model, cfg.trainset, out_dir)
        baseline_df, pnml_df = execute_baseline(cfg, lit_model_h, trainer, loaders_dict)
    else:
        raise ValueError(f'method={cfg.method} is not supported')

    # Save results
    merged = merge_results(baseline_df, pnml_df, cfg)
    merged.to_csv(osp.join(out_dir, 'performance.csv'), index=False)
    logger.info(f"\n{merged[['ood_name', f'AUROC_{cfg.method}', 'AUROC_pnml']]}")
    logger.info('Finish in {:.2f} sec. out_dir={}'.format(time.time() - t0, out_dir))


if __name__ == "__main__":
    run_experiment()
