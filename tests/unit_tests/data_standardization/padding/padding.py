from unittest import TestCase
from unittest.mock import MagicMock

from pandas import DataFrame

from plants_sm.data_standardization.padding_enumerators import PaddingEnumerators
from plants_sm.data_standardization.proteins.padding import SequencePadder


class TestSequencePadding(TestCase):

    def setUp(self) -> None:
        self.dataset = MagicMock()
        self.dataset.dataframe = DataFrame(columns=["sequence", "identifiers"])
        self.dataset.features_dataframe = DataFrame()
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

            "MASLMLSLGSTSLLPREINKDKLKLGTSASNPFLKAKSFSRVTMTVAVKPSRFEGITMAPPDPILGVSEAFKADT"
            "NGMKLNLGVGAYRTEELQPYVLNVVKKAENLMLERGDNKEYLPIEGLAAFNKATAELLFGAGHPVIKEQRVATIQG"
            "LSGTGSLRLAAALIERYFPGAKVVISSPTWGNHKNIFNDAKVPWSEYRYYDPKTIGLDFEGMIADIKEAPEGSFIL"
            "LHGCAHNPTGIDPTPEQWVKIADVIQEKNHIPFFDVAYQGFASGSLDEDAASVRLFAERGMEFFVAQSYSKNLGL"
            "YAERIGAINVVCSSADAATRVKSQLKRIARPMYSNPPVHGARIVANVVGDVTMFSEWKAEMEMMAGRIKTVRQELY"
            "DSLVSKDKSGKDWSFILKQIGMFSFTGLNKAQSDNMTDKWHVYMTKDGRISLAG"
        ]
        self.dataset.dataframe["sequence"] = self.dataset.instances
        self.dataset.dataframe["identifiers"] = self.dataset.identifiers

        self.dataset.dataframe.set_index("identifiers", inplace=True)

    def test_fit(self):
        padder = SequencePadder().fit(self.dataset)

        self.assertEqual(padder.pad_width, len(self.dataset.instances[0]))

    def test_transform(self):
        padder = SequencePadder(padding="left", n_jobs=2).fit(self.dataset)
        padder.transform(self.dataset)

        self.assertEqual(self.dataset.dataframe["sequence"][0], self.dataset.instances[0])
        self.assertEqual(self.dataset.dataframe["sequence"][1],
                         self.dataset.instances[1] +
                         str(PaddingEnumerators.PROTEINS.value) *
                         (len(self.dataset.instances[0]) - len(self.dataset.instances[1])))
        self.assertEqual(len(self.dataset.dataframe["sequence"][1]), len(self.dataset.instances[0]))

        padder = SequencePadder(padding="right", n_jobs=2).fit(self.dataset)
        padder.transform(self.dataset)

        self.assertEqual(self.dataset.dataframe["sequence"][0], self.dataset.instances[0])
        self.assertEqual(self.dataset.dataframe["sequence"][1],
                         str(PaddingEnumerators.PROTEINS.value) *
                         (len(self.dataset.instances[0]) - len(self.dataset.instances[1]))
                         + self.dataset.instances[1]
                         )
        self.assertEqual(len(self.dataset.dataframe["sequence"][1]), len(self.dataset.instances[0]))

        padder = SequencePadder(padding="center", n_jobs=2).fit(self.dataset)
        padder.transform(self.dataset)

        self.assertEqual(self.dataset.dataframe["sequence"][0], self.dataset.instances[0])
        self.assertEqual(
            self.dataset.dataframe["sequence"][1],
            self.dataset.instances[1].center(padder.pad_width, PaddingEnumerators.PROTEINS.value)
        )
        self.assertEqual(len(self.dataset.dataframe["sequence"][1]), len(self.dataset.instances[0]))

    def test_assertions(self):
        padder = SequencePadder(pad_width=5).fit(self.dataset)

        padder.transform(self.dataset)

        self.assertEqual(self.dataset.dataframe["sequence"][0], "-----")

    def test_pad_sequence(self):
        padder = SequencePadder(padding="left").fit(self.dataset)

        pad_dict = padder._pad_sequence(self.dataset.dataframe["sequence"][1], "1")
        self.assertEqual(pad_dict["1"],
                         self.dataset.instances[1] +
                         str(PaddingEnumerators.PROTEINS.value) *
                         (len(self.dataset.instances[0]) - len(self.dataset.instances[1])))
        self.assertEqual(len(pad_dict["1"]), len(self.dataset.instances[0]))

        padder = SequencePadder(padding="right").fit(self.dataset)

        pad_dict = padder._pad_sequence(self.dataset.dataframe["sequence"][1], "1")
        self.assertEqual(pad_dict["1"],
                         str(PaddingEnumerators.PROTEINS.value) *
                         (len(self.dataset.instances[0]) - len(self.dataset.instances[1]))
                         + self.dataset.instances[1])
        self.assertEqual(len(pad_dict["1"]), len(self.dataset.instances[0]))

        padder = SequencePadder(padding="center").fit(self.dataset)
        pad_dict = padder._pad_sequence(self.dataset.dataframe["sequence"][1], "1")
        self.assertEqual(pad_dict["1"],
                         self.dataset.instances[1].center(padder.pad_width, PaddingEnumerators.PROTEINS.value))
        self.assertEqual(len(pad_dict["1"]), len(self.dataset.instances[0]))

    def test_raise_type_error(self):

        with self.assertRaises(ValueError):
            SequencePadder(padding="gyrshujd").fit(self.dataset)
