from plants_sm.ml.featurization.encoding.one_hot_encoder import OneHotEncoder
from plants_sm.ml.tokenisation.compounds.smilespe import AtomLevelTokenizer, SPETokenizer, KmerTokenizer
from unit_tests.featurizer.compounds.test_compound_featurizers import TestCompoundFeaturizers


class TestOneHotEncodings(TestCompoundFeaturizers):

    def test_one_hot_encodings(self):
        one_hot = OneHotEncoder(tokenizer=AtomLevelTokenizer()).fit(self.dataset)
        one_hot.transform(self.dataset)
        self.assertEqual(self.dataset.X().shape, (2, 16, 6))

    def test_one_hot_encodings_SPE(self):
        one_hot = OneHotEncoder(tokenizer=SPETokenizer()).fit(self.dataset)
        one_hot.transform(self.dataset)
        self.assertEqual(self.dataset.X().shape, (2, 7, 9))

    def test_one_hot_encodings_kmer(self):
        one_hot = OneHotEncoder(tokenizer=KmerTokenizer()).fit(self.dataset)
        one_hot.transform(self.dataset)
        self.assertEqual(self.dataset.X().shape, (2, 13, 15))
