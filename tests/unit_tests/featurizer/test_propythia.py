from unittest import TestCase
from unittest.mock import MagicMock

from pandas import DataFrame

from plants_sm.featurization.propythia.propythia import PropythiaWrapper


class TestPropythiaWrapper(TestCase):

    def setUp(self) -> None:
        self.dataset = MagicMock()
        self.dataset.dataframe = DataFrame(columns=["sequence", "identifiers"])
        self.dataset.representation_field = "sequence"
        self.dataset.identifiers = ["0", "1"]
        self.dataset.instances_ids_field = "identifiers"
        self.dataset.features_fields = None
        self.dataset.instances = [
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
        self.dataset.dataframe["sequence"] = self.dataset.instances

    def test_all(self):
        wrapper = PropythiaWrapper("all", n_jobs=2)
        wrapper.featurize(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.dataframe.shape[1], 9598)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_physicochemical(self):
        wrapper = PropythiaWrapper("psycho-chemical", n_jobs=2)
        wrapper.featurize(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.dataframe.shape[1], 28)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_performance(self):
        wrapper = PropythiaWrapper("performance", n_jobs=2)
        wrapper.featurize(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.dataframe.shape[1], 8679)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_aac(self):
        wrapper = PropythiaWrapper("aac", n_jobs=2)
        wrapper.featurize(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.dataframe.shape[1], 8422)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_paac(self):
        wrapper = PropythiaWrapper("paac", n_jobs=2)
        wrapper.featurize(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.dataframe.shape[1], 72)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_auto_correlation(self):
        wrapper = PropythiaWrapper("auto-correlation", n_jobs=2)
        wrapper.featurize(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.dataframe.shape[1], 722)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_composition_transition_distribution(self):
        wrapper = PropythiaWrapper("composition-transition-distribution", n_jobs=2)
        wrapper.featurize(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.dataframe.shape[1], 149)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_seq_order(self):
        wrapper = PropythiaWrapper("seq-order", n_jobs=2)
        wrapper.featurize(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.dataframe.shape[1], 192)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_modlamp_correlation(self):
        wrapper = PropythiaWrapper("modlamp-correlation", n_jobs=2)
        wrapper.featurize(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.dataframe.shape[1], 16)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_modlamp_all(self):
        wrapper = PropythiaWrapper("modlamp-all", n_jobs=2)
        wrapper.featurize(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.dataframe.shape[1], 25)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)


