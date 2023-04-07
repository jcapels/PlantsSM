import os

import pandas as pd

from integration_tests.dataset.test_dataset import TestDataset
from plants_sm.data_standardization.proteins.padding import SequencePadder
from plants_sm.data_standardization.proteins.standardization import ProteinStandardizer
from plants_sm.data_structures.dataset import SingleInputDataset, PLACEHOLDER_FIELD
from plants_sm.data_structures.dataset.multi_input_dataset import MultiInputDataset
from plants_sm.featurization.compounds.deepmol_descriptors import DeepMolDescriptors
from plants_sm.featurization.proteins.propythia.propythia import PropythiaWrapper

from tests import TEST_DIR


class TestLoadInBatches(TestDataset):

    def test_load_in_batches(self):
        dataset = SingleInputDataset.from_csv(os.path.join(TEST_DIR, "data", "proteins.csv"),
                                              batch_size=2,
                                              representation_field="sequence",
                                              instances_ids_field="id")

        self.assertEqual(len(dataset.instances), 1)

        temp_folder = dataset._observer.temporary_folder
        self.assertTrue(os.path.exists(temp_folder.name))
        print(os.listdir(temp_folder.name))
        for folder in os.listdir(temp_folder.name):
            for file in os.listdir(os.path.join(temp_folder.name, folder)):
                for variable_name, _ in dataset.variables_to_save:
                    if variable_name in file:
                        self.assertTrue(os.path.exists(os.path.join(temp_folder.name, folder, file)))

        while dataset.next_batch():
            print(dataset.instances[PLACEHOLDER_FIELD])

    def test_creation_of_temporary_files(self):
        dataset = SingleInputDataset.from_csv(os.path.join(TEST_DIR, "data", "proteins.csv"),
                                              batch_size=2,
                                              representation_field="sequence",
                                              instances_ids_field="id")

        dataset2 = MultiInputDataset.from_csv(self.multi_input_dataset_csv,
                                              representation_field={"proteins": "SEQ",
                                                                     "ligands": "SUBSTRATES"},
                                              instances_ids_field={"interaction": "ids"},
                                              labels_field="LogSpActivity",
                                              batch_size=2)
        temp_folder2 = dataset2._observer.temporary_folder
        temp_folder = dataset._observer.temporary_folder
        self.assertNotEqual(temp_folder.name, temp_folder2.name)

    def test_load_in_batches_multi_dataset(self):
        batch_size = 2
        dataset = MultiInputDataset.from_csv(self.multi_input_dataset_csv,
                                             representation_field={"proteins": "SEQ",
                                                                    "ligands": "SUBSTRATES"},
                                             instances_ids_field={"interaction": "ids"},
                                             labels_field="LogSpActivity",
                                             batch_size=batch_size)

        self.assertEqual(len(dataset.instances), 2)

        temp_folder = dataset._observer.temporary_folder
        self.assertTrue(os.path.exists(temp_folder.name))
        for folder in os.listdir(temp_folder.name):
            for file in os.listdir(os.path.join(temp_folder.name, folder)):
                for variable_name, _ in dataset.variables_to_save:
                    if variable_name in file:
                        self.assertTrue(os.path.exists(os.path.join(temp_folder.name, folder, file)))

        multi_input_dataset = pd.read_csv(self.multi_input_dataset_csv)
        y = []
        while dataset.next_batch():
            y.extend(dataset.y)

        for i, label in enumerate(y):
            self.assertEqual(label, multi_input_dataset["LogSpActivity"][i])

    def test_load_in_batches_multi_dataset_with_padder(self):
        batch_size = 3
        dataset = MultiInputDataset.from_csv(self.multi_input_dataset_csv,
                                             representation_field={"proteins": "SEQ",
                                                                    "ligands": "SUBSTRATES"},
                                             instances_ids_field={"interaction": "ids"},
                                             labels_field="LogSpActivity",
                                             batch_size=batch_size)

        PropythiaWrapper().fit_transform(dataset, "proteins")
        DeepMolDescriptors().fit_transform(dataset, "ligands")
        while dataset.next_batch():
            print(dataset.X)

    def test_protein_padding(self):
        dataset = SingleInputDataset.from_csv(os.path.join(TEST_DIR, "data", "proteins.csv"),
                                              batch_size=2,
                                              representation_field="sequence",
                                              instances_ids_field="id")

        self.assertEqual(len(dataset.instances), 1)

        SequencePadder().fit_transform(dataset)

        while dataset.next_batch():
            print(dataset.instances[PLACEHOLDER_FIELD])

    def test_protein_standardizer(self):
        dataset = SingleInputDataset.from_csv(os.path.join(TEST_DIR, "data", "proteins.csv"),
                                              batch_size=2,
                                              representation_field="sequence",
                                              instances_ids_field="id")

        ProteinStandardizer().fit_transform(dataset)

        while dataset.next_batch():
            print(dataset.instances[PLACEHOLDER_FIELD])

        PropythiaWrapper().fit_transform(dataset)
        while dataset.next_batch():
            print(dataset.instances[PLACEHOLDER_FIELD])
            print(dataset.features[PLACEHOLDER_FIELD])
