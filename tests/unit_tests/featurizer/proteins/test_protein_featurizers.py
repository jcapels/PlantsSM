from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np
import torch
from pandas import DataFrame

from plants_sm.data_structures.dataset.single_input_dataset import PLACEHOLDER_FIELD, SingleInputDataset


class TestProteinFeaturizers(TestCase):

    def X(self):
        return np.array(list(self.dataset.features[PLACEHOLDER_FIELD].values()))

    def add_features(self, instance_type, features):
        self.dataset.features[instance_type] = features

    def setUp(self) -> None:
        self.dataset = MagicMock(spec=SingleInputDataset)
        self.dataset.dataframe = DataFrame(columns=["sequence", "identifiers"])
        self.dataset.features_dataframe = DataFrame()
        self.dataset.representation_field = "sequence"
        self.dataset.identifiers = ["0", "1"]
        self.dataset.instances_ids_field = "identifiers"
        self.dataset.features_fields = {}
        self.dataset.features = {}
        self.dataset.get_instances.return_value = {"0":
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
                                                       "DSLVSKDKSGKDWSFILKQIGMFSFTGLNKAQSDNMTDKWHVYMTKDGRISLAGLSLAKCEYLADAIIDSYERGM"
                                                   }
        self.dataset.dataframe["identifiers"] = self.dataset.identifiers
        torch.seed()
        self.dataset.X = self.X
        self.dataset.add_features = self.add_features
