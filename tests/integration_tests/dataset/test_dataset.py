import os
from unittest import TestCase

import pandas as pd

from plants_sm.data_structures.dataset.multi_input_dataset import MultiInputDataset
from tests import TEST_DIR


class TestDataset(TestCase):

    def setUp(self) -> None:
        self.excel_to_read = os.path.join(TEST_DIR, "data", "drug_list.xlsx")
        self.csv_to_read = os.path.join(TEST_DIR, "data", "proteins.csv")
        self.multi_input_dataset_csv = os.path.join(TEST_DIR, "data", "multi_input_dataset.csv")

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

        # self.multi_input_dataset = MultiInputDataset.from_csv(self.multi_input_dataset_csv,
        #                                                       representation_fields={"proteins": "SEQ",
        #                                                                              "ligands": "SUBSTRATES"},
        #                                                       instances_ids_field={"interaction": "ids"},
        #                                                       labels_field="LogSpActivity")

    def tearDown(self) -> None:
        paths_to_remove = [self.df_path_to_write_csv,
                           self.df_path_to_write_xlsx]

        for path in paths_to_remove:
            if os.path.exists(path):
                os.remove(path)
