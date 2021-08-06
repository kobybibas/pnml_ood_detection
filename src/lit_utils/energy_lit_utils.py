import logging
import os.path as osp

import numpy as np
import torch

from lit_utils.baseline_lit_utils import LitBaseline
from model_utils import get_energy_model
from score_utils import calc_metrics_transformed

logger = logging.getLogger(__name__)


class LitEnergy(LitBaseline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = 1.0

    def get_model(self):
        return get_energy_model(self.model_name, self.ind_name)

    def test_epoch_end(self, outputs):
        probs = torch.vstack([out["probs"] for out in outputs])
        probs_normalized = torch.vstack([out["probs_normalized"] for out in outputs])
        features = torch.vstack([out["features"] for out in outputs])
        logits = torch.vstack([out["logits"] for out in outputs])
        acc = torch.hstack([out["is_correct"] for out in outputs]).float().mean() * 100

        # Omit the samples that a method was fine-tuned on (for instance on ODIN).
        probs = probs[self.validation_size :]
        probs_normalized = probs_normalized[self.validation_size :]
        features = features[self.validation_size :]

        # Compute the normalization factor
        regrets = self.calc_regrets(features, probs_normalized).cpu().numpy()
        max_probs = torch.max(probs, dim=-1).values.cpu().numpy()
        energy = self.temperature * torch.logsumexp(logits, dim=1).cpu().numpy()

        if self.is_ind:
            logger.info("\nValidation set acc {:.2f}%".format(acc))

            # Store IND scores
            self.ind_energy = energy
            self.ind_max_probs = max_probs
            self.ind_regrets = regrets

        else:
            # Run evaluation on the OOD set
            self.baseline_res = calc_metrics_transformed(self.ind_energy, energy)
            self.pnml_res = calc_metrics_transformed(1 - self.ind_regrets, 1 - regrets)

        np.savetxt(osp.join(self.out_dir, f"{self.ood_name}_pnml_regret.txt"), regrets)
        np.savetxt(osp.join(self.out_dir, f"{self.ood_name}_baseline.txt"), max_probs)
