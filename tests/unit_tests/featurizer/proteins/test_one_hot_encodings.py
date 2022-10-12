from unittest import skip

from plants_sm.featurization.one_hot_encoder import OneHotEncoder
from unit_tests.featurizer.proteins.test_protein_featurizers import TestProteinFeaturizers


class TestOneHotEncodings(TestProteinFeaturizers):

    def test_one_hot_encodings_string_as_alphabet(self):
        OneHotEncoder(alphabet="XARNDCEQGHILKMFPSTWYV").fit_transform(self.dataset)

        self.assertEqual(self.dataset.features_dataframe.shape, (906, 21))

    def test_one_hot_encodings_list_as_alphabet(self):
        OneHotEncoder(alphabet=list("XARNDCEQGHILKMFPSTWYV")).fit_transform(self.dataset)

        self.assertEqual(self.dataset.features_dataframe.shape, (906, 21))

    def test_one_hot_encodings_without_alphabet(self):
        OneHotEncoder().fit_transform(self.dataset)

        self.assertEqual(self.dataset.features_dataframe.shape, (906, 20))

    @skip("Not implemented yet")
    def test_with_random_tokens(self):
        OneHotEncoder(alphabet=["AR"]).fit_transform(self.dataset)

        self.assertEqual(self.dataset.features_dataframe.shape, (906, 3))


