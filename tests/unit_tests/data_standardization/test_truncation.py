from unittest import TestCase, skip
from unittest.mock import MagicMock

from pandas import DataFrame

from plants_sm.data_standardization.padding_enumerators import PaddingEnumerators
from plants_sm.data_standardization.proteins.padding import SequencePadder
from plants_sm.data_standardization.truncation import Truncator
from plants_sm.data_structures.dataset import PLACEHOLDER_FIELD, SingleInputDataset


class TestSequencePadding(TestCase):

    def setUp(self) -> None:
        self.dataset = MagicMock(spec=SingleInputDataset)
        self.dataset.dataframe = DataFrame(columns=["sequence", "identifiers"])
        self.dataset.features_dataframe = DataFrame()
        self.dataset.representation_field = "sequence"
        self.dataset.identifiers = ["0", "1"]
        self.dataset.instances_ids_field = "identifiers"
        self.dataset.features_fields = None
        self.dataset.batch_size = None
        self.dataset.instances = {PLACEHOLDER_FIELD: {"0":
                                                          "MASXMLSLGSTSLLPREINKDKLKLGTSASNPFLKAKSFSRVTMTVAVKPSRFEGITMAPPDPILGVSEAFKADT"
                                                          "NGMKLNLGVGAYRTEELQPYVLNVVKKAENLMLERGDNKEYLPIEGLAAFNKATAELLFGAGHPVIKEQRVATIQG"
                                                          "LSGTGSLRLAAALIERYFPGAKVVISSPTWGNHKNIFNDAKVPBSEYRYYDPKTIGLDFEGMIADIKEAPEGSFIL "
                                                          "LHGCXHNPTGIDPTPEQJVKIADVIQEKNHIPFFDVAYQGFASBSLDEDAASVRLFAEROMEFFVAQSYSKNLGL"
                                                          "YAERIGAINVVCSSADAATRVKSQLKRIARPMYSNPPVHGARIVANVVGDUTMFSEWKAEMEMMAGRIKTVRQELY"
                                                          "DSLVSKDKSGKDWSFILKQIGMFSFTGLNKAQSDNMTDKWHVYMTKDGRISLAGLSLAKCEYLADAIIDSYHNVS",
                                                      "1":
                                                          "MASLMLSLGSTSLLPREINKDKLKLGTSASNPFLKAKSFSRVTMTVAVKPSRFEGITMAPPDPILGVSEAFKADT"
                                                          "NGMKLNLGVGAYRTEELQPYVLNVVKKAENLMLERGDNKEYLPIEGLAAFNKATAELLFGAGHPVIKEQRVATIQG"
                                                          "LSGTGSLRLAAALIERYFPGAKVVISSPTWGNHKNIFNDAKVPUSEYRYYDPKTIGLDFEGMIADIKEAPEGSFIL"
                                                          "LHGCAHNPTGIDPTXEQWVKIADVIQEKNHIPFFDVAYQGFASGSLDEDAASVRLFAERGMEFFVAQSYSKNLGL"
                                                          "YAERIGAINVVCSSADAATRVKSQLKRIARPMYZNPPVHGARIVANVVGDVTMFSEWKAEMEMMAGRIKTVRQELY"
                                                          "DSLVSKDKSGKDWSFILKQIGMFSFTGLNKJQSDNMTDKWHVYMTKDGRISLAG"
                                                      }}
        self.dataset.get_instances.return_value = self.dataset.instances[PLACEHOLDER_FIELD]
        self.dataset.dataframe["identifiers"] = self.dataset.identifiers

    def test_sequence_truncation(self):
        Truncator(max_length=100).fit_transform(self.dataset)
        self.assertEqual(len(self.dataset.instances[PLACEHOLDER_FIELD]["0"]), 100)
