from integration_tests.dataset.test_dataset import TestDataset
from plants_sm.data_standardization.x_padder import XPadder
from plants_sm.featurization.proteins.bio_embeddings.esm import ESMEncoder


class TestESMPadding(TestDataset):

    def test_esm_padding(self):
        ESMEncoder(batch_size=1, output_dim=3).fit_transform(self.single_input_dataset)
        self.assertEqual(self.single_input_dataset.X.shape[0], 3)

        XPadder(padding_dimensions=(300, 320)).fit_transform(self.single_input_dataset)

        self.assertEqual(self.single_input_dataset.X.shape[0], 3)
        self.assertEqual(self.single_input_dataset.X[1].shape[0], 300)
        self.assertTrue(all(self.single_input_dataset.X[1][300:] == 0))
