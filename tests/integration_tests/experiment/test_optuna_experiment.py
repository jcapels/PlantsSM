from integration_tests.dataset.test_dataset import TestDataset
from integration_tests.experiment._mock_experiment import MockExperiment


class TestOptunaExperiment(TestDataset):

    def test_experiment(self):
        experiment = MockExperiment(train_dataset=self.single_input_dataset,
                                    validation_dataset=self.single_input_dataset_val)

        experiment.run(n_trials=10)
        experiment.best_experiment.predict(self.single_input_dataset)
        experiment.best_experiment.predict_proba(self.single_input_dataset)



