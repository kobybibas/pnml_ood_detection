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


def execute_odin(cfg, lit_model_h: LitOdin, trainer, loaders_dict) -> (pd.DataFrame, pd.DataFrame):
    # Eval training set (create (x^T x)^{-1} matrix)
    if 'trainset' in loaders_dict:
        train_loader = loaders_dict.pop('trainset')
        trainer.fit(lit_model_h, train_dataloader=train_loader)
    ind_loader = loaders_dict.pop(cfg.trainset)

    epsilons = [0, 0.0005, 0.001, 0.0014, 0.002, 0.0024, 0.005, 0.01, 0.05, 0.1]
    temperatures = [1000]

    results_dict = {}
    for epsilon_num, epsilon in enumerate(epsilons[::-1]):
        results_dict[epsilon] = {}
        for temperature_num, temperature in enumerate(temperatures):

            # Eval out-of-distribution datasets
            baseline_results, pnml_results = [], []
            lit_model_h.set_odin_params(epsilon, temperature)

            # Eval ind score
            lit_model_h.set_ood(lit_model_h.ind_name)
            trainer.test(test_dataloaders=ind_loader, ckpt_path=None, verbose=False)

            for i, (ood_name, ood_dataloader) in enumerate(loaders_dict.items()):
                t0 = time.time()

                # Eval ood score
                lit_model_h.set_ood(ood_name)
                trainer.test(test_dataloaders=ood_dataloader, ckpt_path=None, verbose=False)

                # Get results
                baseline_ood_res, pnml_ood_res = lit_model_h.get_performance()
                baseline_ood_res['ood_name'] = ood_name
                pnml_ood_res['ood_name'] = ood_name

                baseline_results.append(baseline_ood_res)
                pnml_results.append(pnml_ood_res)

                logger.info(
                    '[{}/{}][{}/{}][{}/{}] {}. AUROC [baseline pnml]=[{:.2f} {:.2f}] [eps T]=[{} {}] in {:.1f} sec'.format(
                        epsilon_num, len(epsilons) - 1, temperature_num, len(temperatures) - 1, i,
                                     len(loaders_dict) - 1,
                        ood_name, baseline_ood_res['AUROC'], pnml_ood_res['AUROC'],
                        epsilon, temperature, time.time() - t0))

                results_dict[epsilon][temperature] = {'baseline': baseline_results, 'pnml': pnml_results}

    # Return Dataframes
    baseline_best_results, pnml_best_results = get_best_odin_results(loaders_dict.keys(), results_dict)
    return pd.DataFrame(baseline_best_results), pd.DataFrame(pnml_best_results)


def get_best_odin_results(ood_names: list, results_dict: dict) -> (pd.DataFrame, pd.DataFrame):
    baseline_best_results, pnml_best_results = [], []
    for ood_name in ood_names:

        # Initialize output
        baseline_best_auroc, baseline_best_temperature, baseline_best_epsilon = 0.0, 1.0, 0.0
        pnml_best_auroc, pnml_best_temperature, pnml_best_epsilon = 0.0, 1.0, 0.0

        baseline_ood_best_results, pnml_ood_best_results = None, None

        # Get
        for epsilon, epsilon_dict in results_dict.items():
            for temperature, epsilon_temperature_dict in epsilon_dict.items():

                # Baseline
                for baseline_ood_res in epsilon_temperature_dict['baseline']:
                    if baseline_ood_res['ood_name'] == ood_name:
                        auroc = baseline_ood_res['AUROC']
                        if auroc > baseline_best_auroc:
                            logger.info(f'{ood_name}: baseline best auroc: {pnml_best_auroc}')
                            baseline_ood_best_results = baseline_ood_res
                            baseline_best_auroc = auroc
                            baseline_best_temperature, baseline_best_epsilon = temperature, epsilon

                # pNML
                for pnml_ood_res in epsilon_temperature_dict['pnml']:
                    if pnml_ood_res['ood_name'] == ood_name:
                        auroc = pnml_ood_res['AUROC']
                        if auroc > pnml_best_auroc:
                            logger.info(f'{ood_name}: pnml best auroc: {pnml_best_auroc}')
                            pnml_ood_best_results = pnml_ood_res
                            pnml_best_auroc = auroc
                            pnml_best_temperature, pnml_best_epsilon = temperature, epsilon
        logger.info(ood_name)
        logger.info('baseline: [AUROC epsilon temperature]=[{:.2f} {} {}]'.format(
            baseline_best_auroc, baseline_best_temperature, baseline_best_epsilon))
        logger.info('pnml: [AUROC epsilon temperature]=[{:.2f} {} {}]'.format(
            pnml_best_auroc, pnml_best_epsilon, pnml_best_temperature))
        logger.info()

        baseline_best_results.append(baseline_ood_best_results)
        pnml_best_results.append(pnml_ood_best_results)
    return baseline_best_results, pnml_best_results
