import logging

import torch

from lit_utils.baseline_lit_utils import LitBaseline
from model_utils import get_gram_model
from score_utils import calc_metrics_transformed, split_val_test_idxs, compute_list_of_dict_mean

logger = logging.getLogger(__name__)


class LitGram(LitBaseline):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gram_mins = None
        self.gram_maxs = None
        self.ind_deviations = None
        self.dev_norm_sum = None
        self.model.set_collecting(True)
        self.num_evaluations = 10
        self.ind_features = None
        self.ind_probs_normalized = None

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

        num_layers = len(gram_mins_batches[0][0])
        num_labels = probs.shape[-1]

        self.gram_mins = {c: [None] * num_layers for c in range(num_labels)}
        for c in range(num_labels):
            for l in range(num_layers):
                mins_bathces = [gram_mins_batch[c][l] for gram_mins_batch in gram_mins_batches if c in gram_mins_batch]
                self.gram_mins[c][l] = torch.vstack(mins_bathces).min(dim=0).values

        self.gram_maxs = {c: [None] * num_layers for c in range(num_labels)}
        for c in range(num_labels):
            for l in range(num_layers):
                maxs_batches = [gram_maxs_batch[c][l] for gram_maxs_batch in gram_maxs_batches if c in gram_maxs_batch]
                self.gram_maxs[c][l] = torch.vstack(maxs_batches).max(dim=0).values

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
        acc = torch.hstack([out['is_correct'] for out in outputs]).float().mean() * 100
        deviations = torch.vstack([out['deviations'] for out in outputs])
        features = torch.vstack([out['features'] for out in outputs])
        features = features.to(deviations.device)

        if self.is_ind:
            logger.info('\nValidation set acc {:.2f}%'.format(acc))
            self.ind_features = features
            self.ind_probs_normalized = probs_normalized
            self.ind_deviations = deviations

        else:
            ood_deviations = deviations
            ood_probs_normalized = probs_normalized
            ood_features = features

            baseline_res_list, pnml_res_list = [], []
            for seed in range(1, 1 + self.num_evaluations):
                # Split
                test_indices, val_indices = split_val_test_idxs(len(self.ind_deviations), seed)
                ind_deviations_val = self.ind_deviations[val_indices]
                ind_deviations_test = self.ind_deviations[test_indices]
                ind_features_test = self.ind_features[test_indices]
                ind_probs_normalized_test = self.ind_probs_normalized[test_indices]

                # Compute Gram
                dev_norm_sum = ind_deviations_val.sum(dim=0, keepdims=True) + 10 ** -7
                ind_deviations_norm = (ind_deviations_test / dev_norm_sum).mean(dim=1)
                ood_deviations_norm = (ood_deviations / dev_norm_sum).mean(dim=1)
                baseline_res = calc_metrics_transformed(-ind_deviations_norm.cpu().numpy(),
                                                        -ood_deviations_norm.cpu().numpy())
                print(baseline_res)
                # Compute pNML
                dev_norm_std = ind_deviations_val.std(dim=0, keepdims=True)
                dev_norm_std[dev_norm_std == 0.0] = 1.0
                ind_deviations_norm = torch.sqrt((ind_deviations_test / dev_norm_std).mean(dim=1, keepdims=True))
                ood_deviations_norm = torch.sqrt((ood_deviations / dev_norm_std).mean(dim=1, keepdims=True))

                ind_regrets = self.calc_regrets(ind_features_test * ind_deviations_norm,
                                                ind_probs_normalized_test).cpu().numpy()
                ood_regrets = self.calc_regrets(ood_features * ood_deviations_norm,
                                                ood_probs_normalized).cpu().numpy()
                pnml_res = calc_metrics_transformed(1 - ind_regrets, 1 - ood_regrets)

                # Save this split results
                baseline_res_list.append(baseline_res)
                pnml_res_list.append(pnml_res)

            self.baseline_res = compute_list_of_dict_mean(baseline_res_list)
            self.pnml_res = compute_list_of_dict_mean(pnml_res_list)


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

        # Compute min and max of the gram features (per layer per power) shape=[layers,powers,features]
        mins_c = [layer.min(dim=0).values.cpu() for layer in gram_features_c]
        maxs_c = [layer.max(dim=0).values.cpu() for layer in gram_features_c]

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

    for c in predictions_unique:
        # Initialize per class
        class_idxs = torch.where(c == predictions)[0]
        gram_features_per_class = [gram_feature_layer[class_idxs] for gram_feature_layer in gram_feature_layers]
        max_probs_c = max_probs[predictions == c]

        if c not in gram_mins or c not in gram_maxs:
            logger.warning(f'label {c} is not in training mins and maxs')
            logger.warning(f'gram_mins: {gram_mins.keys()}')
            logger.warning(f'gram_maxs: {gram_maxs.keys()}')

        deviations_c = get_deviations(gram_features_per_class, mins=gram_mins[c], maxs=gram_maxs[c])
        deviations_c /= max_probs_c.to(deviations_c.device).unsqueeze(1)

        deviations.append(deviations_c)

    deviations = torch.cat(deviations, dim=0)

    return deviations


def get_deviations(features_per_layer_list, mins, maxs) -> torch.Tensor:
    deviations = []
    for layer_num, features in enumerate(features_per_layer_list):
        layer_t = features  # [sample,power,value].
        mins_expand = mins[layer_num].unsqueeze(0)
        maxs_expand = maxs[layer_num].unsqueeze(0)

        # Divide each sample by the same min of the layer-power-feature
        layer_t = layer_t.to(mins_expand.device)
        devs_l = (torch.relu(mins_expand - layer_t) / torch.abs(mins_expand + 10 ** -6)).sum(dim=(1, 2))
        devs_l += (torch.relu(layer_t - maxs_expand) / torch.abs(maxs_expand + 10 ** -6)).sum(dim=(1, 2))
        deviations.append(devs_l.unsqueeze(1))
    deviations = torch.cat(deviations, dim=1)  # shape=[num_samples,num_layer]
    return deviations
