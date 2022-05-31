from unittest import TestCase
from unittest.mock import Mock

from pandas import DataFrame

from plants_sm.featurization.propythia import PropythiaWrapper


class TestPropythiaWrapper(TestCase):

    def setUp(self) -> None:
        self.dataset = Mock()
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

    def test_length(self):
        wrapper = PropythiaWrapper("get_length", n_jobs=2)
        wrapper.featurize(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.dataframe.shape[1], 3)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_charge(self):
        wrapper = PropythiaWrapper("get_charge")
        wrapper.featurize(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.dataframe.shape[1], 3)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_charge_density(self):
        wrapper = PropythiaWrapper("get_charge_density")
        wrapper.featurize(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.dataframe.shape[1], 3)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_formula(self):
        wrapper = PropythiaWrapper("get_formula")
        wrapper.featurize(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.dataframe.shape[1], 7)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_bond(self):
        wrapper = PropythiaWrapper("get_bond")
        wrapper.featurize(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.dataframe.shape[1], 6)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_mw(self):
        wrapper = PropythiaWrapper("get_molecular_weight")
        wrapper.featurize(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.dataframe.shape[1], 3)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_gravy(self):
        wrapper = PropythiaWrapper("get_gravy")
        wrapper.featurize(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.dataframe.shape[1], 3)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_aromacity(self):
        wrapper = PropythiaWrapper("get_aromacity")
        wrapper.featurize(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.dataframe.shape[1], 3)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_isoelectric_point(self):
        wrapper = PropythiaWrapper("get_isoelectric_point")
        wrapper.featurize(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.dataframe.shape[1], 3)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_instability_index(self):
        wrapper = PropythiaWrapper("get_instability_index")
        wrapper.featurize(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.dataframe.shape[1], 3)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_secondary_structure(self):
        wrapper = PropythiaWrapper("get_secondary_structure")
        wrapper.featurize(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.dataframe.shape[1], 5)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_molar_extinction_coefficient(self):
        wrapper = PropythiaWrapper("get_molar_extinction_coefficient")
        wrapper.featurize(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.dataframe.shape[1], 4)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_aliphatic_index(self):
        wrapper = PropythiaWrapper("get_aliphatic_index")
        wrapper.featurize(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.dataframe.shape[1], 3)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_boman_index(self):
        wrapper = PropythiaWrapper("get_aliphatic_index")
        wrapper.featurize(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.dataframe.shape[1], 3)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_hydrophobic_ratio(self):
        wrapper = PropythiaWrapper("get_hydrophobic_ratio")
        wrapper.featurize(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.dataframe.shape[1], 3)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_aa_comp(self):
        wrapper = PropythiaWrapper("get_aa_comp")
        wrapper.featurize(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.dataframe.shape[1], 22)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_dp_comp(self):
        wrapper = PropythiaWrapper("get_dp_comp")
        wrapper.featurize(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.dataframe.shape[1], 402)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_tp_comp(self):
        wrapper = PropythiaWrapper("get_tp_comp")
        wrapper.featurize(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.dataframe.shape[1], 8002)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_paac(self):
        wrapper = PropythiaWrapper("get_paac")
        wrapper.featurize(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.dataframe.shape[1], 32)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_paac_p(self):
        wrapper = PropythiaWrapper("get_paac_p")
        wrapper.featurize(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.dataframe.shape[1], 32)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_apaac(self):
        wrapper = PropythiaWrapper("get_apaac")
        wrapper.featurize(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.dataframe.shape[1], 42)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_moreau_broto_auto(self):
        wrapper = PropythiaWrapper("get_moreau_broto_auto")
        wrapper.featurize(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.dataframe.shape[1], 242)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_moran_auto(self):
        wrapper = PropythiaWrapper("get_moran_auto")
        wrapper.featurize(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.dataframe.shape[1], 242)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_geary_auto(self):
        wrapper = PropythiaWrapper("get_geary_auto")
        wrapper.featurize(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.dataframe.shape[1], 242)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_ctd(self):
        wrapper = PropythiaWrapper("get_ctd")
        wrapper.featurize(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.dataframe.shape[1], 149)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_conjoint_triad(self):
        wrapper = PropythiaWrapper("get_conjoint_triad")
        wrapper.featurize(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.dataframe.shape[1], 514)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_socn(self):
        wrapper = PropythiaWrapper("get_socn")
        wrapper.featurize(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.dataframe.shape[1], 92)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_socn_p(self):
        wrapper = PropythiaWrapper("get_socn_p")
        wrapper.featurize(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.dataframe.shape[1], 47)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_qso(self):
        wrapper = PropythiaWrapper("get_qso")
        wrapper.featurize(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.dataframe.shape[1], 102)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_qso_p(self):
        wrapper = PropythiaWrapper("get_qso_p")
        wrapper.featurize(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.dataframe.shape[1], 52)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_calculate_moment(self):
        wrapper = PropythiaWrapper("calculate_moment")
        wrapper.featurize(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.dataframe.shape[1], 3)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_calculate_global(self):
        wrapper = PropythiaWrapper("calculate_global")
        wrapper.featurize(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.dataframe.shape[1], 3)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_calculate_profile(self):
        wrapper = PropythiaWrapper("calculate_profile")
        wrapper.featurize(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.dataframe.shape[1], 4)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_calculate_arc(self):
        wrapper = PropythiaWrapper("calculate_arc")
        wrapper.featurize(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.dataframe.shape[1], 7)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_calculate_autocorrelation(self):
        wrapper = PropythiaWrapper("calculate_autocorrelation")
        wrapper.featurize(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.dataframe.shape[1], 9)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_calculate_crosscorrelation(self):
        wrapper = PropythiaWrapper("calculate_crosscorrelation")
        wrapper.featurize(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.dataframe.shape[1], 9)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_all_physicochemical(self):
        wrapper = PropythiaWrapper("get_all_physicochemical_properties")
        wrapper.featurize(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.dataframe.shape[1], 27)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_all_aac(self):
        wrapper = PropythiaWrapper("get_all_aac")
        wrapper.featurize(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.dataframe.shape[1], 8422)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_all_paac(self):
        wrapper = PropythiaWrapper("get_all_paac")
        wrapper.featurize(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.dataframe.shape[1], 72)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_all_sequence_order(self):
        wrapper = PropythiaWrapper("get_all_sequence_order")
        wrapper.featurize(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.dataframe.shape[1], 192)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_all_correlation(self):
        wrapper = PropythiaWrapper("get_all_correlation")
        wrapper.featurize(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.dataframe.shape[1], 722)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_all_base_class(self):
        wrapper = PropythiaWrapper("get_all_base_class")
        wrapper.featurize(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.dataframe.shape[1], 25)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_all(self):
        wrapper = PropythiaWrapper("get_all", n_jobs=2)
        wrapper.featurize(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.dataframe.shape[1], 2109)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)
