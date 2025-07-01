from unittest import skip

from plants_sm.ml.featurization.encoding.label_encoder import LabelEncoder
from plants_sm.ml.featurization.encoding.one_hot_encoder import OneHotEncoder
from plants_sm.ml.tokenisation.compounds.smilespe import AtomLevelTokenizer, SPETokenizer, KmerTokenizer
from unit_tests.featurizer.compounds.test_compound_featurizers import TestCompoundFeaturizers


class TestLabelEncodings(TestCompoundFeaturizers):

    def test_label_encodings(self):
        one_hot = LabelEncoder(tokenizer=AtomLevelTokenizer()).fit(self.dataset)
        one_hot.transform(self.dataset)
        self.assertEqual(self.dataset.X().shape, (2, 16))

    def test_label_encodings_SPE(self):
        one_hot = LabelEncoder(tokenizer=SPETokenizer()).fit(self.dataset)
        one_hot.transform(self.dataset)
        self.assertEqual(self.dataset.X().shape, (2, 7))

    def test_label_encodings_kmer(self):
        one_hot = LabelEncoder(tokenizer=KmerTokenizer()).fit(self.dataset)
        one_hot.transform(self.dataset)
        self.assertEqual(self.dataset.X().shape, (2, 13))

    @skip("Not implemented yet")
    def test_with_random_tokens(self):
        OneHotEncoder(alphabet=["AR"]).fit_transform(self.dataset)

        self.assertEqual(self.dataset.X().shape, (2, 453, 3))
