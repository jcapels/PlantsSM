from plants_sm.featurization.proteins.encodings.blosum import BLOSSUMEncoder
from plants_sm.featurization.proteins.encodings.nlf import NLFEncoder
from plants_sm.featurization.proteins.encodings.z_scale import ZScaleEncoder
from unit_tests.featurizer.proteins.test_protein_featurizers import TestProteinFeaturizers


class TestEncodings(TestProteinFeaturizers):

    def test_z_scale_encoder(self):
        ZScaleEncoder().fit_transform(self.dataset)

        self.assertEqual(self.dataset.features_dataframe.shape, (906, 5))

    def test_blossum_encoder(self):
        BLOSSUMEncoder().fit_transform(self.dataset)

        self.assertEqual(self.dataset.features_dataframe.shape, (906, 24))

        BLOSSUMEncoder(blosum="blosum50").fit_transform(self.dataset)

        self.assertEqual(self.dataset.features_dataframe.shape, (906, 25))

    def test_nlf(self):
        NLFEncoder().fit_transform(self.dataset)

        self.assertEqual(self.dataset.features_dataframe.shape, (906, 18))
