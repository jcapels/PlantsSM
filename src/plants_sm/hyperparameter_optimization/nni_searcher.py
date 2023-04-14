from nni.experiment import Experiment

from plants_sm.data_structures.dataset import Dataset


class NNISearcher:

    def __init__(self, configuration: dict, models: dict, search_space: dict):
        self.experiment = Experiment('local')
        self.config = configuration
        self._search_space = search_space
        if 'tuner' not in self.config:
            self.config['tuner'] = 'Random'
        self._tuner = self.config['tuner']
        if 'assessor' in self.config:
            self._assessor = self.config['assessor']
        if "max_trial_number" in self.config:
            self.experiment.config.max_trial_number = self.config['max_trial_number']
        else:
            self.experiment.config.max_trial_number = 10
        if "max_experiment_duration" in self.config:
            self.experiment.config.max_experiment_duration = self.config['max_experiment_duration']
        if "trial_concurrency" in self.config:
            self.experiment.config.trial_concurrency = self.config['trial_concurrency']
        self.experiment.config.tuner.name = self._tuner
        self.experiment.config.assessor.name = self._assessor
        self._number_of_trials = self.config['number_of_trials']
        self._models = models

    def search(self, train_dataset: Dataset, validation_dataset: Dataset = None):

        # TODO: the idea is to pickle everything into a temporary folder and then unpickle it in the trial

        for model in self._models:
            search_space = self._search_space[model]
            self.experiment.config.search_space = search_space
            self.experiment.config.trial_command = 'python3 -m plants_sm.hyperparameter_optimization.trial' \
                                                   ' --model {} --train_dataset {} --validation_dataset {}'

            self.experiment.start(8080)

