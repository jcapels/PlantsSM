from integration_tests.dataset.test_dataset import TestDataset
from plants_sm.data_standardization.proteins.padding import SequencePadder
from plants_sm.data_structures.dataset import SingleInputDataset


class TestDatasetSequencePadding(TestDataset):

    def test_dataset_padding(self):
        dataset = SingleInputDataset(dataframe=self.dataframe, representation_field="sequence")

        padder = SequencePadder().fit(dataset)

        self.assertEqual(padder.pad_width, len(dataset.instances[0]))
        self.assertEqual(padder.pad_width, len(dataset.instances[1]))

        padder = SequencePadder(padding="left", n_jobs=2).fit(dataset)
        self.assertEqual(padder.pad_width, len(dataset.instances[0]))
        self.assertEqual(padder.pad_width, len(dataset.instances[1]))

        padder = SequencePadder(padding="center", n_jobs=2).fit(dataset)
        self.assertEqual(padder.pad_width, len(dataset.instances[0]))
        self.assertEqual(padder.pad_width, len(dataset.instances[1]))
