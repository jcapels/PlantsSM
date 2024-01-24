from integration_tests.dataset.test_dataset import TestDataset
from plants_sm.data_standardization.proteins.padding import SequencePadder
from plants_sm.featurization.encoding.one_hot_encoder import OneHotEncoder


class TestOneHotEncoderTokenizer(TestDataset):

    def test_one_hot(self):
        tokenizer = OneHotEncoder()
        tokenizer.fit_transform(self.single_input_dataset)
        for elem in self.single_input_dataset.X[0, 380, :]:
            self.assertEqual(elem, 0)

    def test_one_hot_with_max_length(self):
        tokenizer = OneHotEncoder(max_length=380)
        tokenizer.fit_transform(self.single_input_dataset)

        self.assertTrue(all(self.single_input_dataset.X[0, 350, :] == 0))
        self.assertTrue(any(self.single_input_dataset.X[0, 20, :] == 1))

    def test_one_hot_with_padding(self):
        SequencePadder().fit_transform(self.single_input_dataset)
        tokenizer = OneHotEncoder(padding="-", max_length=380)
        tokenizer.fit_transform(self.single_input_dataset)
        print(self.single_input_dataset.X.shape)
        for elem in self.single_input_dataset.X[0, 350]:
            self.assertEqual(elem, 0)
