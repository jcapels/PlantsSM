from unittest import TestCase
from unittest.mock import MagicMock

from pandas import DataFrame

from plants_sm.data_standardization.proteins.standardization import ProteinStandardizer


class TestStandardizer(TestCase):

    def setUp(self) -> None:
        self.dataset = MagicMock()
        self.dataset.dataframe = DataFrame(columns=["sequence", "identifiers"])
        self.dataset.features_dataframe = DataFrame()
        self.dataset.representation_field = "sequence"
        self.dataset.identifiers = ["0", "1"]
        self.dataset.instances_ids_field = "identifiers"
        self.dataset.features_fields = None
        self.dataset.instances = [
            "MASXMLSLGSTSLLPREINKDKLKLGTSASNPFLKAKSFSRVTMTVAVKPSRFEGITMAPPDPILGVSEAFKADT"
            "NGMKLNLGVGAYRTEELQPYVLNVVKKAENLMLERGDNKEYLPIEGLAAFNKATAELLFGAGHPVIKEQRVATIQG"
            "LSGTGSLRLAAALIERYFPGAKVVISSPTWGNHKNIFNDAKVPBSEYRYYDPKTIGLDFEGMIADIKEAPEGSFIL"
            "LHGCXHNPTGIDPTPEQJVKIADVIQEKNHIPFFDVAYQGFASBSLDEDAASVRLFAEROMEFFVAQSYSKNLGL"
            "YAERIGAINVVCSSADAATRVKSQLKRIARPMYSNPPVHGARIVANVVGDUTMFSEWKAEMEMMAGRIKTVRQELY"
            "DSLVSKDKSGKDWSFILKQIGMFSFTGLNKAQSDNMTDKWHVYMTKDGRISLAGLSLAKCEYLADAIIDSYHNVS",

            "MASLMLSLGSTSLLPREINKDKLKLGTSASNPFLKAKSFSRVTMTVAVKPSRFEGITMAPPDPILGVSEAFKADT"
            "NGMKLNLGVGAYRTEELQPYVLNVVKKAENLMLERGDNKEYLPIEGLAAFNKATAELLFGAGHPVIKEQRVATIQG"
            "LSGTGSLRLAAALIERYFPGAKVVISSPTWGNHKNIFNDAKVPUSEYRYYDPKTIGLDFEGMIADIKEAPEGSFIL"
            "LHGCAHNPTGIDPTXEQWVKIADVIQEKNHIPFFDVAYQGFASGSLDEDAASVRLFAERGMEFFVAQSYSKNLGL"
            "YAERIGAINVVCSSADAATRVKSQLKRIARPMYZNPPVHGARIVANVVGDVTMFSEWKAEMEMMAGRIKTVRQELY"
            "DSLVSKDKSGKDWSFILKQIGMFSFTGLNKJQSDNMTDKWHVYMTKDGRISLAG"
        ]
        self.dataset.dataframe["sequence"] = self.dataset.instances
        self.dataset.dataframe["identifiers"] = self.dataset.identifiers

        self.dataset.dataframe.set_index("identifiers", inplace=True)

    def test_standardizer(self):
        ProteinStandardizer().fit_transform(self.dataset)

        self.assertEqual(self.dataset.dataframe["sequence"][0].count("B"), 0)
        self.assertEqual(self.dataset.dataframe["sequence"][0].count("Z"), 0)
        self.assertEqual(self.dataset.dataframe["sequence"][0].count("X"), 0)
        self.assertEqual(self.dataset.dataframe["sequence"][0].count("J"), 0)
        self.assertEqual(self.dataset.dataframe["sequence"][0].count("U"), 0)
        self.assertEqual(self.dataset.dataframe["sequence"][0].count("O"), 0)

        self.assertEqual(self.dataset.dataframe["sequence"][1].count("B"), 0)
        self.assertEqual(self.dataset.dataframe["sequence"][1].count("Z"), 0)
        self.assertEqual(self.dataset.dataframe["sequence"][1].count("X"), 0)
        self.assertEqual(self.dataset.dataframe["sequence"][1].count("J"), 0)
        self.assertEqual(self.dataset.dataframe["sequence"][1].count("U"), 0)
        self.assertEqual(self.dataset.dataframe["sequence"][1].count("O"), 0)
