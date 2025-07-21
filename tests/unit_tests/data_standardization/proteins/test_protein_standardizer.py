from unittest import TestCase
from unittest.mock import MagicMock

from pandas import DataFrame

from plants_sm.data_standardization.proteins.standardization import ProteinStandardizer
from plants_sm.data_structures.dataset import SingleInputDataset, PLACEHOLDER_FIELD


class TestStandardizer(TestCase):

    def setUp(self) -> None:

        self.dataset = MagicMock(spec=SingleInputDataset)
        self.dataset.dataframe = DataFrame(columns=["sequence", "identifiers"])
        self.dataset.features_dataframe = DataFrame()
        self.dataset.representation_field = "sequence"
        self.dataset.identifiers = ["0", "1"]
        self.dataset.instances_ids_field = "identifiers"
        self.dataset.features_fields = None
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
        self.dataset.batch_size = None

    def test_standardizer(self):
        ProteinStandardizer().fit_transform(self.dataset)

        instances = list(self.dataset.instances[PLACEHOLDER_FIELD].values())

        self.assertEqual(instances[0].count("B"), 0)
        self.assertEqual(instances[0].count("Z"), 0)
        self.assertEqual(instances[0].count("X"), 0)
        self.assertEqual(instances[0].count("J"), 0)
        self.assertEqual(instances[0].count("U"), 0)
        self.assertEqual(instances[0].count("O"), 0)

        self.assertEqual(instances[1].count("B"), 0)
        self.assertEqual(instances[1].count("Z"), 0)
        self.assertEqual(instances[1].count("X"), 0)
        self.assertEqual(instances[1].count("J"), 0)
        self.assertEqual(instances[1].count("U"), 0)
        self.assertEqual(instances[1].count("O"), 0)
