import os
from unittest import TestCase

import pandas as pd

from plants_sm.data_structures.dataset import PandasDataset
from plants_sm.featurization.propythia import PropythiaWrapper

from tests import TEST_DIR


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
        propythia = PropythiaWrapper(descriptor="get_length")
        propythia.featurize(dataset)

        self.assertEqual(dataset.dataframe.shape[0], 2)
        self.assertEqual(dataset.dataframe.shape[1], 3)
        self.assertEqual(dataset.features_fields, ["length"])
        self.assertEqual(dataset.instances_ids_field, "identifier")

    def test_with_real_data(self):
        dataset_path = os.path.join(TEST_DIR, "data", "data_athaliana.csv")
        dataset = PandasDataset(labels_field=["union_class", "Aracyc class"], representation_field="sequence_test",
                                instances_ids_field="gene")
        dataset.from_csv(dataset_path)
        propythia = PropythiaWrapper(descriptor="get_all_physicochemical_properties", n_jobs=1)
        propythia.featurize(dataset)
        self.assertEqual(len(dataset.features_fields), 25)
