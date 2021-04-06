import logging

import numpy as np
import torch
from tqdm import tqdm

from lit_utils.baseline_lit_utils import LitBaseline
from model_utils import get_gram_model
from score_utils import calc_metrics_transformed

logger = logging.getLogger(__name__)


class LitGram(LitBaseline):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gram_mins = None
        self.gram_maxs = None
        self.ind_deviations = None
        self.dev_norm_sum = None
        self.model.set_collecting(True)

    def get_model(self):
        return get_gram_model(self.model_name, self.ind_name)

    def training_step(self, batch, batch_idx):
        output = super(LitGram, self).training_step(batch, batch_idx)

        # Get gram features
        gram_features_batch = self.model.collect_gram_features()

        gram_mins, gram_maxs = compute_minmaxs_train(gram_features_batch, output['probs'])
        output['gram_mins'] = gram_mins
        output['gram_maxs'] = gram_maxs
        return output

    def training_epoch_end(self, outputs):
        super().training_epoch_end(outputs)

        # Combine mins and maxs from batches
        probs = torch.vstack([out['probs'] for out in outputs])
        gram_mins_batches = [out['gram_mins'] for out in outputs]
        gram_maxs_batches = [out['gram_maxs'] for out in outputs]

        num_layers = len(gram_mins_batches[0])
        num_labels = probs.shape[-1]

        self.gram_mins = {c: [] * num_layers for c in range(num_labels)}
        for c in range(num_labels):
            for l in range(num_layers):
                mins_bathces = [gram_mins_batch[c][l] for gram_mins_batch in gram_mins_batches if c in gram_mins_batch]
                self.gram_mins[c][l] = torch.vstack(mins_bathces).min(dim=0).values

        self.gram_maxs = {c: [] * num_layers for c in range(num_labels)}
        for c in range(num_labels):
            for l in range(num_layers):
                maxs_bathces = [gram_maxs_batch[c][l] for gram_maxs_batch in gram_maxs_batches if c in gram_maxs_batch]
                self.gram_maxs[c][l] = torch.vstack(maxs_bathces).max(dim=0).values

    def test_step(self, batch, batch_idx):
        output = super().test_step(batch, batch_idx)

        # Get gram features
        gram_features_batch = self.model.collect_gram_features()

        # Compute deviations of the test gram features from the min and max of the trainset
        deviations = compute_deviations(gram_features_batch, output['probs'], self.gram_mins, self.gram_maxs)
        output['deviations'] = deviations
        return output

    def test_epoch_end(self, outputs):
        probs = torch.vstack([out['probs'] for out in outputs])
        probs_normalized = torch.vstack([out['probs_normalized'] for out in outputs])
        features = torch.vstack([out['features'] for out in outputs])
        acc = torch.hstack([out['is_correct'] for out in outputs]).float().mean() * 100
        deviations = torch.hstack([out['deviations'] for out in outputs]).float().mean() * 100

        # Compute the normalization factor
        regrets = self.calc_regrets(features, probs_normalized)
        max_probs = torch.max(probs, dim=-1).values

        if self.is_ind:
            logger.info('\nValidation set acc {:.2f}%'.format(acc))

            # Store IND scores
            self.ind_max_probs = max_probs.cpu().numpy()
            self.ind_regrets = regrets.cpu().numpy()
            self.ind_deviations = deviations.cpu().numpy()

        else:
            # Run evaluation on the OOD set
            self.pnml_res = calc_metrics_transformed(1 - self.ind_regrets,
                                                     (1 - regrets).cpu().numpy())

            # Compute Gram
            self.dev_norm_sum = deviations.sum(axis=0, keepdims=True) + 10 ** -7
            self.ind_deviations = -(deviations / self.dev_norm_sum).mean(dim=1)
            deviations = -(deviations / self.dev_norm_sum).mean(dim=1)
            self.baseline_res = calc_metrics_transformed(-self.ind_deviations,
                                                     deviations.cpu().numpy())


def compute_minmaxs_train(gram_feature_layers, probs):
    predictions = torch.argmax(probs, dim=1)

    predictions_unique = torch.unique(predictions)
    predictions_unique.sort()
    predictions_unique = predictions_unique.tolist()

    # Initialize outputs
    mins, maxs = {}, {}

    # Iterate on labels
    for c in predictions_unique:
        # Extract samples that are predicted as c
        class_idxs = torch.where(c == predictions)[0]
        gram_features_c = [gram_feature_in_layer_i[class_idxs] for gram_feature_in_layer_i in gram_feature_layers]

        # compute min and max of the gram features (per layer per power) shape=[layers,powers,features]
        mins_c = [layer.min(dim=0).values for layer in gram_features_c]
        maxs_c = [layer.max(dim=0).values for layer in gram_features_c]

        # Save
        mins[c] = mins_c
        maxs[c] = maxs_c

    return mins, maxs


def compute_deviations(gram_feature_layers, probs, gram_mins, gram_maxs):
    # Initialize outputs
    deviations = []

    max_probs, predictions = probs.max(dim=1)  # [values, idxs]

    # Iterate on labels
    predictions_unique = torch.unique(predictions)
    predictions_unique.sort()
    predictions_unique = predictions_unique.tolist()

    for c in tqdm(predictions_unique, desc='compute_deviations'):
        # Initialize per class
        class_idxs = torch.where(c == predictions)[0]
        gram_features_per_class = [gram_feature_layer[class_idxs] for gram_feature_layer in gram_feature_layers]
        max_probs_c = max_probs[predictions == c]

        if c not in gram_mins or c not in gram_maxs:
            logger.warning(f'label {c} is not in training mins and maxs')
            logger.warning(f'gram_mins: {gram_mins.keys()}')
            logger.warning(f'gram_maxs: {gram_maxs.keys()}')

        deviations_c = get_deviations(gram_features_per_class, mins=gram_mins[c], maxs=gram_maxs[c])
        deviations_c /= max_probs_c.unsqueeze(1)  # max_probs_c[:, np.newaxis]

        deviations.append(deviations_c)

    deviations = torch.cat(deviations, dim=0)

    return deviations


def relu(x):
    return x * (x > 0)


def get_deviations(features_per_layer_list, mins, maxs) -> np.ndarray:
    deviations = []
    for layer_num, features in enumerate(features_per_layer_list):
        layer_t = features  # features.transpose(2, 0, 1)  # [sample,power,value].
        mins_expand = mins[layer_num].unsqueeze(0)  # np.expand_dims(mins[layer_num], axis=0)
        maxs_expand = maxs[layer_num].unsqueeze(0)  # np.expand_dims(maxs[layer_num], axis=0)

        # Divide each sample by the same min of the layer-power-feature
        devs_l = (relu(mins_expand - layer_t) / torch.abs(mins_expand + 10 ** -6)).sum(axis=(1, 2))
        devs_l += (relu(layer_t - maxs_expand) / torch.abs(maxs_expand + 10 ** -6)).sum(axis=(1, 2))
        deviations.append(devs_l.unsqueeze(1))
        # deviations.append(np.expand_dims(devs_l, 1))
    deviations = torch.cat(deviations, dim=1)  # shape=[num_samples,num_layer]
    return deviations
