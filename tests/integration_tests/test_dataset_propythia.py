from unittest import TestCase

import pandas as pd

from plants_sm.data_structures.dataset import PandasDataset
from plants_sm.featurization.proteins.propythia.propythia import PropythiaWrapper


class TestDatasetPropythia(TestCase):

    def setUp(self) -> None:
        self.dataframe = pd.DataFrame(columns=["sequence"])
        self.sequences = [
            "MASLMLSLGSTSLLPREINKDKLKLGTSASNPFLKAKSFSRVTMTVAVKPSRFEGITMAPPDPILGVSEAFKADT"
            "NGMKLNLGVGAYRTEELQPYVLNVVKKAENLMLERGDNKEYLPIEGLAAFNKATAELLFGAGHPVIKEQRVATIQG"
            "LSGTGSLRLAAALIERYFPGAKVVISSPTWGNHKNIFNDAKVPWSEYRYYDPKTIGLDFEGMIADIKEAPEGSFIL"
            "LHGCAHNPTGIDPTPEQWVKIADVIQEKNHIPFFDVAYQGFASGSLDEDAASVRLFAERGMEFFVAQSYSKNLGL"
            "YAERIGAINVVCSSADAATRVKSQLKRIARPMYSNPPVHGARIVANVVGDVTMFSEWKAEMEMMAGRIKTVRQELY"
            "DSLVSKDKSGKDWSFILKQIGMFSFTGLNKAQSDNMTDKWHVYMTKDGRISLAGLSLAKCEYLADAIIDSYHNVS",

            "MGEVAVDPAFIQAVEHRPNLTIIQAEGIPIIDLSSLINSSSSSDDSPKSDELERLISEIGKACSDWGFFQ"
            "VINHGVPLECRQRIESVSRKFFALSKEEKLKVKRDEENPLGYYDTEHTKNVRDWKEVFDFTVQNPAVIPA"
            "SDEPEDEDVRKINSQWPQYPPEFREACEDYVKEMEKLSFKLLELISLSLGLPANRMNRFFEKDETSFIRL"
            "NHYPPCPISHLALGVGRHKDAGALTVLAQDDVGGLQVKRKIDGEWILVKPTPNAYIINVGDIIQVWSNDK"
            "YESVEHRVMVNSEKERFSIPFFFNPAHYTWVEPLKELINQQNPSKYKAYNWGKFFTTRKGSNFKKLDVEN"
            "IQIYHFKNI"
        ]
        self.dataframe["sequence"] = self.sequences

    def test_create_dataset_propythia(self):
        dataset = PandasDataset(dataframe=self.dataframe, representation_field="sequence")
        propythia = PropythiaWrapper(descriptor="all")
        propythia.featurize(dataset)

        self.assertEqual(dataset.dataframe.shape[0], 2)
        self.assertEqual(dataset.dataframe.shape[1], 9598)
        self.assertEqual(dataset.features_fields[0], "length")
        self.assertEqual(dataset.instances_ids_field, "identifier")
