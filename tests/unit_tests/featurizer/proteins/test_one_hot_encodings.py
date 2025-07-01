from unittest import skip

from plants_sm.ml.featurization.encoding.label_encoder import LabelEncoder
from plants_sm.ml.featurization.encoding.one_hot_encoder import OneHotEncoder
from unit_tests.featurizer.proteins.test_protein_featurizers import TestProteinFeaturizers


class TestOneHotEncodings(TestProteinFeaturizers):

    def test_one_hot_encodings_string_as_alphabet(self):
        OneHotEncoder(alphabet="XARNDCEQGHILKMFPSTWYV").fit_transform(self.dataset)

        self.assertEqual(self.dataset.X().shape, (2, 453, 21))

    def test_one_hot_encodings_list_as_alphabet(self):
        OneHotEncoder(alphabet=list("XARNDCEQGHILKMFPSTWYV")).fit_transform(self.dataset)

        self.assertEqual(self.dataset.X().shape, (2, 453, 21))

    def test_one_hot_encodings_without_alphabet(self):
        one_hot = OneHotEncoder().fit(self.dataset)
        one_hot.transform(self.dataset)

        self.assertEqual(self.dataset.X().shape, (2, 453, 20))
        for i in range(self.dataset.X().shape[1]):
            amino_acid = self.dataset.get_instances()["0"][i]

            for j in range(len(self.dataset.X()[0, i, :])):
                try:
                    self.assertEqual(one_hot.tokens[amino_acid][j], self.dataset.X()[0, i, j])
                except AssertionError:
                    print("Amino acid: ", amino_acid)
                    print("Index: ", i)
                    print("One hot: ", one_hot.tokens[amino_acid])
                    print("Dataset: ", self.dataset.X()[0, i, :])
                    raise AssertionError

    def test_one_hot_encodings_2d(self):
        LabelEncoder().fit_transform(self.dataset)

        self.assertEqual(self.dataset.X().shape, (2, 453))
        print(self.dataset.X())

    @skip("Not implemented yet")
    def test_with_random_tokens(self):
        OneHotEncoder(alphabet=["AR"]).fit_transform(self.dataset)

        self.assertEqual(self.dataset.X().shape, (2, 453, 3))



