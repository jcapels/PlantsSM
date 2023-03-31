import os
from unittest import TestCase

import pandas as pd

from plants_sm.data_structures.dataset import SingleInputDataset
from plants_sm.data_structures.dataset.multi_input_dataset import MultiInputDataset
from plants_sm.featurization.compounds.deepmol_descriptors import DeepMolDescriptors
from plants_sm.featurization.proteins.bio_embeddings.word2vec import Word2Vec
from tests import TEST_DIR


class TestDataset(TestCase):

    def setUp(self) -> None:
        self.excel_to_read = os.path.join(TEST_DIR, "data", "drug_list.xlsx")
        self.csv_to_read = os.path.join(TEST_DIR, "data", "proteins.csv")
        self.multi_input_dataset_csv = os.path.join(TEST_DIR, "data", "multi_input_dataset.csv")
        self.single_input_dataset_csv = os.path.join(TEST_DIR, "data", "proteins.csv")

        self.df_path_to_write_csv = os.path.join(TEST_DIR, "data", "test.csv")
        self.df_path_to_write_xlsx = os.path.join(TEST_DIR, "data", "test.xlsx")
        self.dataframe = pd.DataFrame(columns=["ids", "sequence"])
        self.sequences = [
            "MASLMLSLGSTSLLPREINKDKLKLGTSASNPFLKAKSFSRVTMTVAVKPSRFEGITMAPPDPILGVSEAFKADT"
            "NGMKLNLGVGAYRTEELQPYVLNVVKKAENLMLERGDNKEYLPIEGLAAFNKATAELLFGAGHPVIKEQRVATIQG"
            "LSGTGSLRLAAALIERYFPGAKVVISSPTWGNHKNIFNDAKVPWSEYRYYDPKTIGLDFEGMIADIKEAPEGSFIL"
            "LHGCAHNPTGIDPTPEQWVKIADVIQEKNHIPFFDVAYQGFASGSLDEDAASVRLFAERGMEFFVAQSYSKNLGL"
            "YAERIGAINVVCSSADAATRVKSQLKRIARPMYSNPPVHGARIVANVVGDVTMFSEWKAEMEMMAGRIKTVRQELY"
            "DSLVSKDKSGKDWSFILKQIGMFSFTGLNKAQSDNMTDKWHVYMTKDGRISLAGLSLAKCEYLADAIIDSYHNVS",

            "MASLMLSLGSTSLLPREINKDKLKLGTSASNPFLKAKSFSRVTMTVAVKPSRFEGITMAPPDPILGVSEAFKADT"
            "NGMKLNLGVGAYRTEELQPYVLNVVKKAENLMLERGDNKEYLPIEGLAAFNKATAELLFGAGHPVIKEQRVATIQG"
            "LSGTGSLRLAAALIERYFPGAKVVISSPTWGNHKNIFNDAKVPWSEYRYYDPKTIGLDFEGMIADIKEAPEGSFIL"
            "LHGCAHNPTGIDPTPEQWVKIADVIQEKNHIPFFDVAYQGFASGSLDEDAASVRLFAERGMEFFVAQSYSKNLGL"
            "YAERIGAINVVCSSADAATRVKSQLKRIARPMYSNPPVHGARIVANVVGDVTMFSEWKAEMEMMAGRIKTVRQELY"
            "DSLVSKDKSGKDWSFILKQIGMFSFTGLNKAQSDNMTDKWHVYMTKDGRISLAGLSLAKCEYLADAIIDWSFILK"
        ]
        self.dataframe["sequence"] = self.sequences
        self.dataframe["ids"] = ["WP_003399745.1", "WP_003399671.1"]

        self.compounds_dataframe = pd.DataFrame(columns=["ids", "sequence"])
        self.sequences = [
            "CCCCCCCO", "C1=CC=C(C=C1)C=O"
        ]
        self.compounds_dataframe["sequence"] = self.sequences
        self.compounds_dataframe["ids"] = ["WP_003399745.1", "WP_003399671.1"]

        self.multi_input_dataset = MultiInputDataset.from_csv(self.multi_input_dataset_csv,
                                                              representation_fields={"proteins": "SEQ",
                                                                                     "ligands": "SUBSTRATES"},
                                                              instances_ids_field={"interaction": "ids"},
                                                              labels_field="LogSpActivity")

        self.single_input_dataset = SingleInputDataset.from_csv(self.single_input_dataset_csv,
                                                                representation_field="sequence",
                                                                instances_ids_field="id",
                                                                labels_field="y")

        self.single_input_dataset_val = SingleInputDataset.from_csv(self.single_input_dataset_csv,
                                                                    representation_field="sequence",
                                                                    instances_ids_field="id",
                                                                    labels_field="y")

    def test_read_single_input_and_write_to_csv(self):
        """
        Test the read and write to csv methods.
        """

        Word2Vec().fit_transform(self.multi_input_dataset, "proteins")
        DeepMolDescriptors().fit_transform(self.multi_input_dataset, "ligands")

        self.multi_input_dataset.to_csv("test.csv", index=False)

        written_dataset = pd.read_csv("test.csv")

        actual_dataset = pd.read_csv(self.multi_input_dataset_csv)

        self.assertTrue(written_dataset.SEQ.equals(actual_dataset.SEQ))
        self.assertTrue("word2vec_511" in list(written_dataset.columns))

        # remove the file
        os.remove("test.csv")

    def test_read_multi_input_and_write_to_csv(self):
        """
        Test the read and write to csv methods.
        """

        dataset = SingleInputDataset.from_csv(self.multi_input_dataset_csv, representation_field="sequence",
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

    def tearDown(self) -> None:
        paths_to_remove = [self.df_path_to_write_csv,
                           self.df_path_to_write_xlsx]

        for path in paths_to_remove:
            if os.path.exists(path):
                os.remove(path)
