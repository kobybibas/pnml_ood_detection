import logging

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from model_utils import get_model
from score_utils import calc_metrics_transformed

logger = logging.getLogger(__name__)


class LitBaseline(pl.LightningModule):

    def __init__(self, model_name: str, ind_name: str):
        super().__init__()
        self.model_name = model_name
        self.ind_name = ind_name
        self.model = self.get_model()
        self.x_t_x_inv = None
        self.w = self.model.fc if hasattr(self.model, 'fc') else self.linear

        # IND
        self.is_ind = True
        self.ind_max_probs = None
        self.ind_regrets = None

        # OOD
        self.ood_name = ''

        # Metrics
        self.baseline_res, self.pnml_res = None, None

        self.validation_size = 0

    def set_validation_size(self, validation_size: int):
        self.validation_size = validation_size
        logger.info(f'set_validation_size: validation_size={self.validation_size}')

    def set_ood(self, ood_name: str):
        self.ood_name = ood_name
        self.is_ind = ood_name == self.ind_name
        logger.info(f'set_ood: ood_name={self.ood_name} is_ind={self.is_ind}')

    def get_model(self):
        return get_model(self.model_name, self.ind_name)

    def configure_optimizers(self):
        # We don't want to train, set lr to 0 and gives optimizer different param to optimize on
        return torch.optim.Adam(self.model.parameters(), lr=0.0)

    def backward(self, loss, optimizer, optimizer_idx, *args, **kwargs):
        pass

    def forward(self, x):
        self.model.eval()

        with torch.no_grad():
            logits = self.model(x)
            features = self.model.get_features()
            features = features / torch.linalg.norm(features, dim=-1, keepdim=True)

            # Forward with feature normalization
            logits_w_norm_features = self.w(features)
        return logits, features, logits_w_norm_features

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits, features, logits_w_norm_features = self.forward(x)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        probs_normalized = torch.nn.functional.softmax(logits_w_norm_features, dim=-1)
        loss = F.cross_entropy(logits, y)

        y_hat = probs.argmax(dim=-1)
        is_correct = y_hat == y
        output = {'probs': probs, 'probs_normalized': probs_normalized,
                  'features': features, 'loss': loss, 'is_correct': is_correct}
        return output

    def training_epoch_end(self, outputs):
        features = torch.vstack([out['features'] for out in outputs])
        acc = torch.hstack([out['is_correct'] for out in outputs]).float().mean() * 100
        logger.info('\nTraining set acc {:.2f}%'.format(acc))

        # Calc regrets
        x_t_x = torch.matmul(features.t(), features)
        _, s, _ = torch.linalg.svd(x_t_x, compute_uv=False)
        logger.info(f'Training set smallest singular values: {s[-10:]}')
        self.x_t_x_inv = torch.linalg.inv(x_t_x) if s[-1] > 1e-16 else torch.linalg.pinv(x_t_x, hermitian=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits, features, logits_w_norm_features = self.forward(x)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        probs_normalized = torch.nn.functional.softmax(logits_w_norm_features, dim=-1)

        y_hat = probs.argmax(dim=-1)
        is_correct = y_hat == y

        output = {'probs': probs, 'probs_normalized': probs_normalized, 'is_correct': is_correct,
                  'features': features}
        return output

    def test_epoch_end(self, outputs):
        probs = torch.vstack([out['probs'] for out in outputs])
        probs_normalized = torch.vstack([out['probs_normalized'] for out in outputs])
        features = torch.vstack([out['features'] for out in outputs])
        acc = torch.hstack([out['is_correct'] for out in outputs]).float().mean() * 100

        # Omit the samples that a method was fine-tuned on (for instance on ODIN).
        probs = probs[self.validation_size:]
        probs_normalized = probs_normalized[self.validation_size:]
        features = features[self.validation_size:]

        # Compute the normalization factor
        regrets = self.calc_regrets(features, probs_normalized)
        max_probs = torch.max(probs, dim=-1).values

        if self.is_ind:
            logger.info('\nValidation set acc {:.2f}%'.format(acc))

            # Store IND scores
            self.ind_max_probs = max_probs.cpu().numpy()
            self.ind_regrets = regrets.cpu().numpy()

        else:
            # Run evaluation on the OOD set
            self.baseline_res = calc_metrics_transformed(self.ind_max_probs,
                                                         max_probs.cpu().numpy())
            self.pnml_res = calc_metrics_transformed(1 - self.ind_regrets,
                                                     (1 - regrets).cpu().numpy())

    def get_performance(self):
        return self.baseline_res, self.pnml_res

    def calc_regrets(self, features: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        x_proj = torch.matmul(torch.matmul(features.unsqueeze(1), self.x_t_x_inv), features.unsqueeze(-1))
        x_proj = x_proj.squeeze(-1)
        x_t_g = x_proj / (1 + x_proj)

        # compute the normalization factor
        n_classes = probs.shape[-1]
        nf = torch.sum(probs / (probs + (1 - probs) * (probs ** x_t_g)), dim=-1)
        regrets = torch.log(nf) / torch.log(torch.tensor(n_classes))
        return regrets
