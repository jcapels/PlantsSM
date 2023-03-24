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

        temp_folder = dataset._observers[0].temporary_folder
        self.assertTrue(os.path.exists(temp_folder.name))
        print(os.listdir(temp_folder.name))
        for folder in os.listdir(temp_folder.name):
            for file in os.listdir(os.path.join(temp_folder.name, folder)):
                for variable_name, _ in dataset.variables_to_save:
                    if variable_name in file:
                        self.assertTrue(os.path.exists(os.path.join(temp_folder.name, folder, file)))

