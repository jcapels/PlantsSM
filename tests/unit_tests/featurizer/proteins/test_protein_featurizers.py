from unittest import TestCase
from unittest.mock import MagicMock

from pandas import DataFrame


class TestProteinFeaturizers(TestCase):

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
        self.dataset.dataframe["identifiers"] = self.dataset.identifiers