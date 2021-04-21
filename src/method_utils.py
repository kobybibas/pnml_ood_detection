import logging
import time

import pandas as pd

from lit_utils.baseline_lit_utils import LitBaseline
from lit_utils.odin_lit_utils import LitOdin

logger = logging.getLogger(__name__)


def execute_baseline(cfg, lit_model_h: LitBaseline, trainer, loaders_dict) -> (pd.DataFrame, pd.DataFrame):
    # Eval training set (create (x^T x)^{-1} matrix)
    train_loader = loaders_dict.pop('trainset')
    trainer.fit(lit_model_h, train_dataloader=train_loader)

    # Get in-distribution scores
    ind_loader = loaders_dict.pop(cfg.trainset)
    trainer.test(test_dataloaders=ind_loader, ckpt_path=None, verbose=False)

    # Eval out-of-distribution datasets
    baseline_results, pnml_results = [], []
    for i, (ood_name, ood_dataloader) in enumerate(loaders_dict.items()):
        t0 = time.time()

        # Evaluate ood score
        lit_model_h.set_ood(ood_name)
        trainer.test(test_dataloaders=ood_dataloader, ckpt_path=None, verbose=False)
        baseline_res, pnml_res = lit_model_h.get_performance()

        baseline_res['ood_name'] = ood_name
        pnml_res['ood_name'] = ood_name

        logger.info('[{}/{}] {} as ood. AUROC [baseline pnml]=[{:.2f} {:.2f}] in {:.1f} sec'.format(
            i, len(loaders_dict) - 1, ood_name, baseline_res['AUROC'], pnml_res['AUROC'], time.time() - t0))

        # Save
        baseline_results.append(baseline_res)
        pnml_results.append(pnml_res)

    return pd.DataFrame(baseline_results), pd.DataFrame(pnml_results)


def execute_odin(cfg, lit_model_h: LitOdin, trainer, loaders_dict, odin_params: dict) -> (pd.DataFrame, pd.DataFrame):
    # Eval training set (create (x^T x)^{-1} matrix)
    if 'trainset' in loaders_dict:
        train_loader = loaders_dict.pop('trainset')
        trainer.fit(lit_model_h, train_dataloader=train_loader)
    ind_loader = loaders_dict.pop(cfg.trainset)

    # Eval out-of-distribution datasets
    baseline_results, pnml_results = [], []
    for i, (ood_name, ood_dataloader) in enumerate(loaders_dict.items()):
        t0 = time.time()

        lit_model_h.set_odin_params(odin_params[ood_name]['epsilon'],
                                    odin_params[ood_name]['temperature'])

        # Eval ind score
        lit_model_h.set_ood(lit_model_h.ind_name)
        trainer.test(test_dataloaders=ind_loader, ckpt_path=None, verbose=False)

        # Eval ood score
        lit_model_h.set_ood(ood_name)
        trainer.test(test_dataloaders=ood_dataloader, ckpt_path=None, verbose=False)

        # Get results
        baseline_res, pnml_res = lit_model_h.get_performance()
        baseline_res['ood_name'] = ood_name
        pnml_res['ood_name'] = ood_name

        baseline_results.append(baseline_res)
        pnml_results.append(pnml_res)

        logger.info('[{}/{}] {} as ood. AUROC [baseline pnml]=[{:.2f} {:.2f}] in {:.1f} sec'.format(
            i, len(loaders_dict) - 1, ood_name, baseline_res['AUROC'], pnml_res['AUROC'], time.time() - t0))

    # Insert back the ind loader
    loaders_dict[cfg.trainset] = ind_loader

    # Return Dataframes
    return pd.DataFrame(baseline_results), pd.DataFrame(pnml_results)
