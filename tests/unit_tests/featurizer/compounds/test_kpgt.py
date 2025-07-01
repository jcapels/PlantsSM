from unittest import skip
from plants_sm.featurization.compounds.kpgt import KPGP
from unit_tests.featurizer.compounds.test_compound_featurizers import TestCompoundFeaturizers

@skip("Skip because it takes too long to download KPGT model")
class TestKPGT(TestCompoundFeaturizers):

    def test_kpgt_download(self):
        KPGP().fit(self.dataset)