import os
from unittest import TestCase

from plants_sm.data_structures.dataset import SingleInputDataset, PLACEHOLDER_FIELD
from plants_sm.io import read_csv

from tests import TEST_DIR


class TestLoadInBatches(TestCase):

    def test_load_in_batches(self):
        def load_in_batches():
            return read_csv(os.path.join(TEST_DIR, "data", "proteins.csv"), chunksize=1)

        dataset = SingleInputDataset.from_csv(os.path.join(TEST_DIR, "data", "proteins.csv"), batch_size=1,
                                              representation_field="sequence", instances_ids_field="id")
        self.assertEqual(len(dataset.instances), 1)
        instance1 = list(dataset.instances[PLACEHOLDER_FIELD].values())[0]

        next(dataset)
        self.assertEqual(len(dataset.instances), 1)
        instance2 = list(dataset.instances[PLACEHOLDER_FIELD].values())[0]

        self.assertNotEqual(instance2, instance1)

