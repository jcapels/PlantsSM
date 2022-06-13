import unittest
from unittest import TestCase
from unittest.mock import Mock

from pandas import DataFrame

from plants_sm.featurization.bio_embeddings.unirep import UniRepEmbeddings


class TestEmbeddings(TestCase):

    def setUp(self) -> None:
        self.dataset = Mock()
        self.dataset.dataframe = DataFrame(columns=["sequence", "identifiers"])
        self.dataset.representation_field = "sequence"
        self.dataset.identifiers = ["0", "1"]
        self.dataset.instances_ids_field = "identifiers"
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

    @unittest.skip("embeddings use cuda")
    def test_unirep_embeddings(self):
        embeddings = UniRepEmbeddings(device="0")
        embeddings.featurize(self.dataset)
