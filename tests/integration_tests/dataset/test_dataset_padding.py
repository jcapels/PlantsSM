from integration_tests.dataset.test_dataset import TestDataset
from plants_sm.ml.data_standardization.proteins.padding import SequencePadder
from plants_sm.ml.data_structures.dataset import SingleInputDataset


class TestDatasetSequencePadding(TestDataset):

    def test_dataset_padding(self):
        dataset = SingleInputDataset(dataframe=self.dataframe, representation_field="sequence")

        padder = SequencePadder().fit(dataset)

        instances = list(dataset.get_instances().values())

        self.assertEqual(padder.pad_width, len(instances[0]))
        self.assertEqual(padder.pad_width, len(instances[1]))

        padder = SequencePadder(padding="left", n_jobs=2).fit(dataset)
        instances = list(dataset.get_instances().values())
        self.assertEqual(padder.pad_width, len(instances[0]))
        self.assertEqual(padder.pad_width, len(instances[1]))

        padder = SequencePadder(padding="center", n_jobs=2).fit(dataset)
        instances = list(dataset.get_instances().values())
        self.assertEqual(padder.pad_width, len(instances[0]))
        self.assertEqual(padder.pad_width, len(instances[1]))
