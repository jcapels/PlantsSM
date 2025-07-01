from plants_sm.ml.featurization.compounds.chembert2a import ChemBert2a
from unit_tests.featurizer.compounds.test_compound_featurizers import TestCompoundFeaturizers


class TestChemBert2a(TestCompoundFeaturizers):

    def test_chembert2a(self):
        ChemBert2a().fit_transform(self.dataset)
        assert self.dataset.X().shape == (2, 600)