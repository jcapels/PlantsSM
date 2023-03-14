import logging
import os
import shutil
from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np
import torch
from pandas import DataFrame

from plants_sm.data_structures.dataset import SingleInputDataset, PLACEHOLDER_FIELD

from tests import TEST_DIR


class TestModels(TestCase):

    def X(self):
        return np.array(list(self.train_dataset.features[PLACEHOLDER_FIELD]))

    def setUp(self) -> None:
        logging.getLogger('tensorflow').setLevel(logging.ERROR)

        self.train_dataset = MagicMock(spec=SingleInputDataset)
        self.train_dataset.dataframe = DataFrame(columns=["sequence", "identifiers"])
        self.train_dataset.features_dataframe = DataFrame()
        self.train_dataset.representation_field = "sequence"
        self.train_dataset.identifiers = ["0", "1", "2", "3"]
        self.train_dataset.y = np.array([0, 1, 0, 1])
        self.train_dataset.instances_ids_field = "identifiers"
        self.train_dataset.features_fields = {}
        self.train_dataset.features = {}
        self.train_dataset.get_instances.return_value = {"0":
                                                             "MASLMLSLGSTSLLPREINKDKLKLGTSASNPFLKAKSFSRVTMTVAVKPSRFEGITMAPPDPILGVSEAFKADT"
                                                             "NGMKLNLGVGAYRTEELQPYVLNVVKKAENLMLERGDNKEYLPIEGLAAFNKATAELLFGAGHPVIKEQRVATIQG"
                                                             "LSGTGSLRLAAALIERYFPGAKVVISSPTWGNHKNIFNDAKVPWSEYRYYDPKTIGLDFEGMIADIKEAPEGSFIL"
                                                             "LHGCAHNPTGIDPTPEQWVKIADVIQEKNHIPFFDVAYQGFASGSLDEDAASVRLFAERGMEFFVAQSYSKNLGL"
                                                             "YAERIGAINVVCSSADAATRVKSQLKRIARPMYSNPPVHGARIVANVVGDVTMFSEWKAEMEMMAGRIKTVRQELY"
                                                             "DSLVSKDKSGKDWSFILKQIGMFSFTGLNKAQSDNMTDKWHVYMTKDGRISLAGLSLAKCEYLADAIIDSYHNVS",
                                                         "1":
                                                             "MASLMLSLGSTSLLPREINKDKLKLGTSASNPFLKAKSFSRVTMTVAVKPSRFEGITMAPPDPILGVSEAFKADT"
                                                             "NGMKLNLGVGAYRTEELQPYVLNVVKKAENLMLERGDNKEYLPIEGLAAFNKATAELLFGAGHPVIKEQRVATIQG"
                                                             "LSGTGSLRLAAALIERYFPGAKVVISSPTWGNHKNIFNDAKVPWSEYRYYDPKTIGLDFEGMIADIKEAPEGSFIL"
                                                             "LHGCAHNPTGIDPTPEQWVKIADVIQEKNHIPFFDVAYQGFASGSLDEDAASVRLFAERGMEFFVAQSYSKNLGL"
                                                             "YAERIGAINVVCSSADAATRVKSQLKRIARPMYSNPPVHGARIVANVVGDVTMFSEWKAEMEMMAGRIKTVRQELY"
                                                             "DSLVSKDKSGKDWSFILKQIGMFSFTGLNKAQSDNMTDKWHVYMTKDGRISLAGLSLAKCEYLADAIIDSYERGM",

                                                         "2": "MASLMLSLGSTSLLPREINKDKLKLGTSASNPFLKAKSFSRVTMTVAVKPSRFEGITMAPPDPILGVSEAFKADT"
                                                              "NGMKLNLGVGAYRTEELQPYVLNVVKKAENLMLERGDNKEYLPIEGLAAFNKATAELLFGAGHPVIKEQRVATIQG"
                                                              "LSGTGSLRLAAALIERYFPGAKVVISSPTWGNHKNIFNDAKVPLGSTSLLPRWSEYRYYDPKTIGLDFEGMIADIKE"
                                                              "APEGSFIL"
                                                              "LHGCAHNPTGIDPTPEQWVKIADVIQEKNHIPFFDVAYQGFASGSLDEDAASVRLFAERGMEFFVAQSYSKNLGL"
                                                              "YAERIGAINVVCSSADAATRVKSQLKRIARPMYSNPPVHGARIVANVVGDVTMFSEWKAEMEMMAGRIKTVRQELY"
                                                              "DSLVSKDKSGKDWSFILKQIGMFSFTGLNKAQSDNMTDKWHVYMTKDGRISLAGLSLAKCEYLADAIIDSYERGM",
                                                         "3": "MASLMLSLGSTSLLPREINKDKLKLGTSASNPFLKAKSFSRVTMTVAVKPSRFEGITMAPPDPILGVSEAFKADT"
                                                              "NGMKLNLGVGAYRTEELQPYVLNVVKKAENLMLERGDNKEYLPIEGLAAFNKATAELLFGAGHPVIKEQRVATIQG"
                                                              "LSGTGSLRLAAALIERYFPGAKVVISSPTWGNHKNIFNDAKVPLGSTSLLPRWSEYRYYDPKTIGLDFEGMIADIKE"
                                                              "ELQPYVLNVVKKAENLMLERGDN"
                                                              "LHGCAHNPTGIDPTPEQWVKIADVIQEKNHIPFFDVAYQGFASGSLDEDAASVRLFAERGMEFFVAQSYSKNLGL"
                                                         }
        self.train_dataset.dataframe["identifiers"] = self.train_dataset.identifiers
        torch.seed()
        self.train_dataset.features = {PLACEHOLDER_FIELD:
                                           np.random.randint(0, 100, size=(len(self.train_dataset.identifiers), 100))}
        self.train_dataset.X = self.X()

        self.validation_dataset = MagicMock(spec=SingleInputDataset)
        self.validation_dataset.dataframe = DataFrame(columns=["sequence", "identifiers"])
        self.validation_dataset.features_dataframe = DataFrame()
        self.validation_dataset.representation_field = "sequence"
        self.validation_dataset.identifiers = ["0", "1", "2", "3"]
        self.validation_dataset.y = np.array([0, 1, 0, 1])
        self.validation_dataset.instances_ids_field = "identifiers"
        self.validation_dataset.features_fields = {}
        self.validation_dataset.features = {}
        self.validation_dataset.get_instances.return_value = {"0":
                                                                  "MASLMLSLGSTSLLPREINKDKLKLGTSASNPFLKAKSFSRVTMTVAVKPSRFEGITMAPPDPILGVSEAFKADT"
                                                                  "NGMKLNLGVGAYRTEELQPYVLNVVKKAENLMLERGDNKEYLPIEGLAAFNKATAELLFGAGHPVIKEQRVATIQG"
                                                                  "LSGTGSLRLAAALIERYFPGAKVVISSPTWGNHKNIFNDAKVPWSEYRYYDPKTIGLDFEGMIADIKEAPEGSFIL"
                                                                  "LHGCAHNPTGIDPTPEQWVKIADVIQEKNHIPFFDVAYQGFASGSLDEDAASVRLFAERGMEFFVAQSYSKNLGL"
                                                                  "YAERIGAINVVCSSADAATRVKSQLKRIARPMYSNPPVHGARIVANVVGDVTMFSEWKAEMEMMAGRIKTVRQELY"
                                                                  "DSLVSKDKSGKDWSFILKQIGMFSFTGLNKAQSDNMTDKWHVYMTKDGRISLAGLSLAKCEYLADAIIDSYHNVS",
                                                              "1":
                                                                  "MASLMLSLGSTSLLPREINKDKLKLGTSASNPFLKAKSFSRVTMTVAVKPSRFEGITMAPPDPILGVSEAFKADT"
                                                                  "NGMKLNLGVGAYRTEELQPYVLNVVKKAENLMLERGDNKEYLPIEGLAAFNKATAELLFGAGHPVIKEQRVATIQG"
                                                                  "LSGTGSLRLAAALIERYFPGAKVVISSPTWGNHKNIFNDAKVPWSEYRYYDPKTIGLDFEGMIADIKEAPEGSFIL"
                                                                  "LHGCAHNPTGIDPTPEQWVKIADVIQEKNHIPFFDVAYQGFASGSLDEDAASVRLFAERGMEFFVAQSYSKNLGL"
                                                                  "YAERIGAINVVCSSADAATRVKSQLKRIARPMYSNPPVHGARIVANVVGDVTMFSEWKAEMEMMAGRIKTVRQELY"
                                                                  "DSLVSKDKSGKDWSFILKQIGMFSFTGLNKAQSDNMTDKWHVYMTKDGRISLAGLSLAKCEYLADAIIDSYERGM",

                                                              "2": "MASLMLSLGSTSLLPREINKDKLKLGTSASNPFLKAKSFSRVTMTVAVKPSRFEGITMAPPDPILGVSEAFKADT"
                                                                   "NGMKLNLGVGAYRTEELQPYVLNVVKKAENLMLERGDNKEYLPIEGLAAFNKATAELLFGAGHPVIKEQRVATIQG"
                                                                   "LSGTGSLRLAAALIERYFPGAKVVISSPTWGNHKNIFNDAKVPLGSTSLLPRWSEYRYYDPKTIGLDFEGMIADIKE"
                                                                   "APEGSFIL"
                                                                   "LHGCAHNPTGIDPTPEQWVKIADVIQEKNHIPFFDVAYQGFASGSLDEDAASVRLFAERGMEFFVAQSYSKNLGL"
                                                                   "YAERIGAINVVCSSADAATRVKSQLKRIARPMYSNPPVHGARIVANVVGDVTMFSEWKAEMEMMAGRIKTVRQELY"
                                                                   "DSLVSKDKSGKDWSFILKQIGMFSFTGLNKAQSDNMTDKWHVYMTKDGRISLAGLSLAKCEYLADAIIDSYERGM",
                                                              "3": "MASLMLSLGSTSLLPREINKDKLKLGTSASNPFLKAKSFSRVTMTVAVKPSRFEGITMAPPDPILGVSEAFKADT"
                                                                   "NGMKLNLGVGAYRTEELQPYVLNVVKKAENLMLERGDNKEYLPIEGLAAFNKATAELLFGAGHPVIKEQRVATIQG"
                                                                   "LSGTGSLRLAAALIERYFPGAKVVISSPTWGNHKNIFNDAKVPLGSTSLLPRWSEYRYYDPKTIGLDFEGMIADIKE"
                                                                   "ELQPYVLNVVKKAENLMLERGDN"
                                                                   "LHGCAHNPTGIDPTPEQWVKIADVIQEKNHIPFFDVAYQGFASGSLDEDAASVRLFAERGMEFFVAQSYSKNLGL"
                                                              }
        self.validation_dataset.dataframe["identifiers"] = self.train_dataset.identifiers
        torch.seed()
        self.validation_dataset.features = {PLACEHOLDER_FIELD:
                                                np.random.randint(0, 100,
                                                                  size=(len(self.train_dataset.identifiers), 100))}
        self.validation_dataset.X = self.X()

        self.path_to_save = os.path.join(TEST_DIR, "data", "test_save_model")

    def tearDown(self) -> None:
        if os.path.exists(self.path_to_save):
            shutil.rmtree(self.path_to_save)
