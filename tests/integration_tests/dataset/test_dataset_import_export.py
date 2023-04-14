import os
import shutil

import pandas as pd

from integration_tests.dataset.test_dataset import TestDataset
from plants_sm.data_standardization.proteins.padding import SequencePadder
from plants_sm.data_structures.dataset import SingleInputDataset
from plants_sm.data_structures.dataset.multi_input_dataset import MultiInputDataset
from plants_sm.featurization.compounds.deepmol_descriptors import DeepMolDescriptors
from plants_sm.featurization.encoding.one_hot_encoder import OneHotEncoder
from plants_sm.featurization.proteins.bio_embeddings.word2vec import Word2Vec


class TestDatasetImportExport(TestDataset):

    def test_read_multi_input_and_write_to_csv(self):
        """
        Test the read and write to csv methods.
        """

        multi_input_dataset = MultiInputDataset.from_csv(self.multi_input_dataset_csv,
                                                              representation_field={"proteins": "SEQ",
                                                                                    "ligands": "SUBSTRATES"},
                                                              instances_ids_field={"interaction": "ids"},
                                                              labels_field="LogSpActivity")

        Word2Vec().fit_transform(multi_input_dataset, "proteins")
        DeepMolDescriptors().fit_transform(multi_input_dataset, "ligands")

        multi_input_dataset.to_csv("test.csv", index=False)

        written_dataset = pd.read_csv("test.csv")

        actual_dataset = pd.read_csv(self.multi_input_dataset_csv)

        self.assertTrue(written_dataset.SEQ.equals(actual_dataset.SEQ))
        # remove the file
        os.remove("test.csv")

    def test_read_multi_input_and_write_to_csv_3d_features(self):
        """
        Test the read and write to csv methods.
        """

        SequencePadder().fit_transform(self.multi_input_dataset, instance_type="proteins")
        Word2Vec(output_shape_dimension=3).fit_transform(self.multi_input_dataset, "proteins")
        DeepMolDescriptors().fit_transform(self.multi_input_dataset, "ligands")

        self.multi_input_dataset.to_csv("test.csv", index=False)
        self.assertTrue(os.path.exists("test.pkl"))

        written_dataset = pd.read_csv("test.csv")

        actual_dataset = pd.read_csv(self.multi_input_dataset_csv)

        self.assertNotEqual(list(written_dataset.SEQ), list(actual_dataset.SEQ))

        # remove the file
        os.remove("test.csv")
        os.remove("test.pkl")

    def test_read_single_input_and_write_to_csv_3d_features(self):
        """
        Test the read and write to csv methods.
        """

        SequencePadder().fit_transform(self.single_input_dataset)
        Word2Vec(output_shape_dimension=3).fit_transform(self.single_input_dataset)

        self.single_input_dataset.to_csv("test.csv", index=False)
        self.assertTrue(os.path.exists("test.pkl"))

        written_dataset = pd.read_csv("test.csv")

        actual_dataset = pd.read_csv(self.single_input_dataset_csv)

        self.assertNotEqual(list(written_dataset.sequence), list(actual_dataset.sequence))

        # remove the file
        os.remove("test.csv")
        os.remove("test.pkl")

    def test_read_single_input_and_write_to_csv(self):
        """
        Test the read and write to csv methods.
        """

        dataset = SingleInputDataset.from_csv(self.single_input_dataset_csv,
                                              representation_field="sequence",
                                              instances_ids_field="id",
                                              labels_field="y")

        Word2Vec().fit_transform(dataset)

        dataset.to_csv("test.csv", index=False)

        written_dataset = pd.read_csv("test.csv")

        actual_dataset = pd.read_csv(self.single_input_dataset_csv)

        self.assertTrue(written_dataset.sequence.equals(actual_dataset.sequence))
        self.assertTrue("word2vec_511" in list(written_dataset.columns))

        # remove the file
        os.remove("test.csv")

    def test_read_and_load_single_input_dataset(self):
        dataset = SingleInputDataset.from_csv(self.single_input_dataset_csv,
                                              representation_field="sequence",
                                              instances_ids_field="id",
                                              labels_field="y")

        Word2Vec().fit_transform(dataset)
        dataset.save("test")

        loaded_dataset = SingleInputDataset.load("test.pkl")
        self.assertTrue(loaded_dataset.representation_field == "sequence")
        self.assertTrue(loaded_dataset.instances_ids_field == "id")
        self.assertTrue(loaded_dataset.dataframe.equals(dataset.dataframe))

        os.remove("test.pkl")

        dataset = SingleInputDataset.from_csv(self.single_input_dataset_csv,
                                              representation_field="sequence",
                                              instances_ids_field="id",
                                              labels_field="y",
                                              batch_size=2)

        Word2Vec().fit_transform(dataset)
        dataset.save("test")

        loaded_dataset = SingleInputDataset.load("test")
        self.assertTrue(loaded_dataset.representation_field == "sequence")
        self.assertTrue(loaded_dataset.instances_ids_field == "id")
        self.assertTrue(loaded_dataset.dataframe.equals(dataset.dataframe))

        shutil.rmtree("test")

        loaded_dataset.save_features("test")

        dataset = SingleInputDataset.from_csv(self.single_input_dataset_csv,
                                              representation_field="sequence",
                                              instances_ids_field="id",
                                              labels_field="y",
                                              batch_size=2)

        dataset.load_features("test")
        dataset.next_batch()
        self.assertTrue(dataset.X.shape == (2, 512))
        dataset.next_batch()
        self.assertTrue(dataset.X.shape == (1, 512))
        shutil.rmtree("test")

    def test_read_and_load_multi_input_dataset(self):
        dataset = MultiInputDataset.from_csv(self.multi_input_dataset_csv,
                                             representation_field={"proteins": "SEQ",
                                                                   "ligands": "SUBSTRATES"},
                                             instances_ids_field={"interaction": "ids"},
                                             labels_field="LogSpActivity")

        Word2Vec().fit_transform(dataset, "proteins")
        dataset.save("test")

        loaded_dataset = MultiInputDataset.load("test.pkl")
        self.assertTrue(loaded_dataset.representation_field == {"proteins": "SEQ",
                                                                "ligands": "SUBSTRATES"})
        self.assertTrue(loaded_dataset.dataframe.equals(dataset.dataframe))

        os.remove("test.pkl")

        dataset = MultiInputDataset.from_csv(self.multi_input_dataset_csv,
                                             representation_field={"proteins": "SEQ",
                                                                   "ligands": "SUBSTRATES"},
                                             instances_ids_field={"interaction": "ids"},
                                             labels_field="LogSpActivity",
                                             batch_size=2)

        Word2Vec().fit_transform(dataset, "proteins")
        dataset.save("test")

        loaded_dataset = MultiInputDataset.load("test")
        self.assertTrue(loaded_dataset.representation_field == {"proteins": "SEQ",
                                                                "ligands": "SUBSTRATES"})
        self.assertTrue(loaded_dataset.dataframe.equals(dataset.dataframe))

        shutil.rmtree("test")

        OneHotEncoder().fit_transform(dataset, "ligands")

        loaded_dataset.save_features("test")

        dataset = MultiInputDataset.from_csv(self.multi_input_dataset_csv,
                                             representation_field={"proteins": "SEQ",
                                                                   "ligands": "SUBSTRATES"},
                                             instances_ids_field={"interaction": "ids"},
                                             labels_field="LogSpActivity",
                                             batch_size=2)

        dataset.load_features("test")

        dataset.next_batch()
        print(dataset.X["proteins"].shape)
        self.assertTrue(dataset.X["proteins"].shape == (2, 512))
        dataset.next_batch()
        self.assertTrue(dataset.X["proteins"].shape == (2, 512))
        dataset.next_batch()
        self.assertTrue(dataset.X["ligands"].shape == (2, 27, 10))

        shutil.rmtree("test")
