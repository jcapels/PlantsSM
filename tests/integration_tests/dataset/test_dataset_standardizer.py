from unittest import TestCase

import pandas as pd

from plants_sm.ml.data_standardization.proteins.standardization import ProteinStandardizer
from plants_sm.ml.data_structures.dataset import SingleInputDataset


class TestDatasetStandardizer(TestCase):

    def setUp(self):
        self.dataframe = pd.DataFrame(columns=["ids", "sequence"])
        self.sequences = [
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
        self.dataframe["sequence"] = self.sequences
        self.dataframe["ids"] = ["WP_003399745.1", "WP_003399671.1"]
        self.dataset = SingleInputDataset(dataframe=self.dataframe, representation_field="sequence")

    def test_dataset_standardizer(self):

        ProteinStandardizer().fit_transform(self.dataset)
        instances = list(self.dataset.get_instances().values())
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