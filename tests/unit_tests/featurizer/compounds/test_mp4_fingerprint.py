from unittest import skip

from plants_sm.data_structures.dataset import PLACEHOLDER_FIELD
from unit_tests.featurizer.compounds.test_compound_featurizers import TestCompoundFeaturizers


# @skip("Requires conda")
class TestMP4Fingerprint(TestCompoundFeaturizers):

    def test_mp4_fingerprint(self):
        from plants_sm.featurization.compounds.map4_fingerprint import MAP4Fingerprint
        mp4 = MAP4Fingerprint()
        mp4._fit(self.dataset, PLACEHOLDER_FIELD)
        features = mp4._featurize(self.dataset.instances[0])
        self.assertEqual(features.shape[0], 1024)
