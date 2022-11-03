from unittest import skip

from plants_sm.featurization.encoding.one_hot_encoder import OneHotEncoder
from plants_sm.tokenisation.compounds.smilespe import AtomLevelTokenizer, SPETokenizer, KmerTokenizer
from unit_tests.featurizer.compounds.test_compound_featurizers import TestCompoundFeaturizers


class TestOneHotEncodings(TestCompoundFeaturizers):

    def test_one_hot_encodings_2d(self):
        one_hot = OneHotEncoder(output_shape_dimension=2, tokenizer=AtomLevelTokenizer()).fit(self.dataset)
        one_hot.transform(self.dataset)
        print(self.dataset.X())
        print(one_hot.tokens)
        self.assertEqual(self.dataset.X().shape, (2, 16))

    def test_one_hot_encodings_2d_SPE(self):
        one_hot = OneHotEncoder(output_shape_dimension=2, tokenizer=SPETokenizer()).fit(self.dataset)
        one_hot.transform(self.dataset)
        print(self.dataset.X())
        print(one_hot.tokens)
        self.assertEqual(self.dataset.X().shape, (2, 7))

    def test_one_hot_encodings_2d_kmer(self):
        one_hot = OneHotEncoder(output_shape_dimension=2, tokenizer=KmerTokenizer()).fit(self.dataset)
        one_hot.transform(self.dataset)
        print(self.dataset.X())
        print(one_hot.tokens)
        self.assertEqual(self.dataset.X().shape, (2, 13))


    @skip("Not implemented yet")
    def test_with_random_tokens(self):
        OneHotEncoder(alphabet=["AR"]).fit_transform(self.dataset)

        self.assertEqual(self.dataset.X().shape, (2, 453, 3))



