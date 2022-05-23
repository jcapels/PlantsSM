from unittest import TestCase
from unittest.mock import Mock

from pandas import DataFrame

from plants_sm.featurization.propythia import PropythiaWrapper


class TestPropythiaWrapper(TestCase):

    def test_propythia_wrapper(self):
        dataset = Mock()
        dataset.dataframe = DataFrame()
        dataset.instances = ["MASLMLSLGSTSLLPREINKDKLKLGTSASNPFLKAKSFSRVTMTVAVKPSRFEGITMAPPDPILGVSEAFKADT"
                             "NGMKLNLGVGAYRTEELQPYVLNVVKKAENLMLERGDNKEYLPIEGLAAFNKATAELLFGAGHPVIKEQRVATIQG"
                             "LSGTGSLRLAAALIERYFPGAKVVISSPTWGNHKNIFNDAKVPWSEYRYYDPKTIGLDFEGMIADIKEAPEGSFIL"
                             "LHGCAHNPTGIDPTPEQWVKIADVIQEKNHIPFFDVAYQGFASGSLDEDAASVRLFAERGMEFFVAQSYSKNLGL"
                             "YAERIGAINVVCSSADAATRVKSQLKRIARPMYSNPPVHGARIVANVVGDVTMFSEWKAEMEMMAGRIKTVRQELY"
                             "DSLVSKDKSGKDWSFILKQIGMFSFTGLNKAQSDNMTDKWHVYMTKDGRISLAGLSLAKCEYLADAIIDSYHNVS"] \
                            * 2

        wrapper = PropythiaWrapper("get_aa_index1", name="KRIW790103")

        wrapper.featurize(dataset)

        wrapper = PropythiaWrapper("get_bin_resi_prop")

        wrapper.featurize(dataset)
        self.assertEqual(dataset.dataframe.shape[0], 2)
        self.assertEqual(dataset.dataframe.shape[1], 11345)
