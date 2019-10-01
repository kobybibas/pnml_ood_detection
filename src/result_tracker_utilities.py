import json
import os
import pathlib
import time


class ResultTracker:
    def __init__(self, experiment_name: str, output_root: str):
        """
        Initialize logger class
        :param experiment_name: the experiment type- use for saving string of the outputs/
        :param output_root: the directory to which the output will be saved.
        """

        self.json_file_name = None
        self.results_dict = {}

        self.unique_time = time.strftime("%Y%m%d_%H%M%S")
        self.output_folder = os.path.join(output_root, '%s_%s' %
                                          (experiment_name, self.unique_time))
        pathlib.Path(self.output_folder).mkdir(parents=True, exist_ok=True)
        self.define_json_output(os.path.join(self.output_folder, 'results_%s_%s.json' %
                                             (experiment_name, self.unique_time)))

    def define_json_output(self, json_file_name: str):
        """
        set the output json file name. The results of the PNML will be save into.
        :param json_file_name: the file name of the results file
        :return:
        """
        self.json_file_name = json_file_name

    def save_json_file(self):
        """
        Save results into hard disk
        :return:
        """
        with open(self.json_file_name, 'w') as outfile:
            json.dump(self.results_dict,
                      outfile,
                      sort_keys=True,
                      indent=4,
                      ensure_ascii=False)

    def add_entry_to_results_dict(self, test_idx_sample: int, prob_key_str: str, prob: list,
                                  train_loss: float, test_loss: float):
        """
        Add results entry into the result dict
        :param test_idx_sample: the test sample index in the testset.
        :param prob_key_str: the label of the test sample that the model was trained with.
        :param prob: the predicted probability assignment.
        :param train_loss: the loss of the trainset after training the model with the test sample
        :param test_loss: the loss of the testset after training the model with the test sample
        :return:
        """
        if str(test_idx_sample) not in self.results_dict:
            self.results_dict[str(test_idx_sample)] = {}

        self.results_dict[str(test_idx_sample)][prob_key_str] = {}
        self.results_dict[str(test_idx_sample)][prob_key_str]['prob'] = prob
        self.results_dict[str(test_idx_sample)][prob_key_str]['train_loss'] = train_loss
        self.results_dict[str(test_idx_sample)][prob_key_str]['test_loss'] = test_loss

    def add_org_prob_to_results_dict(self, test_idx_sample: int, prob_org: list, true_label: int):
        """
        Adding the ERM base model probability assignment on the test sample.
        :param test_idx_sample: the test sample index in the testset.
        :param prob_org: the predicted probability assignment
        :param true_label: the true label of the test sample
        :return:
        """
        if str(test_idx_sample) not in self.results_dict:
            self.results_dict[str(test_idx_sample)] = {}

        self.results_dict[str(test_idx_sample)]['original'] = {}
        self.results_dict[str(test_idx_sample)]['original']['prob'] = prob_org
        self.results_dict[str(test_idx_sample)]['true_label'] = int(true_label)
