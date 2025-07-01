from plants_sm.ml.data_structures.dataset import PLACEHOLDER_FIELD
from plants_sm.ml.featurization.compounds.deepmol_descriptors import DeepMolDescriptors
from unit_tests.featurizer.compounds.test_compound_featurizers import TestCompoundFeaturizers


class TestDeepMolDescriptors(TestCompoundFeaturizers):

    def test_deepmol_wrapper(self):
        wrapper = DeepMolDescriptors(n_jobs=2)
        wrapper._fit(self.dataset, PLACEHOLDER_FIELD)
        features = wrapper._featurize(self.dataset.instances[PLACEHOLDER_FIELD]['0'])
        self.assertEqual(features.shape[0], 2048)

    def test_morgan_fingerprints_wrapper(self):
        wrapper = DeepMolDescriptors(n_jobs=2, preset="morgan_fingerprints", kwargs={"size": 1024})
        wrapper._fit(self.dataset, PLACEHOLDER_FIELD)
        features = wrapper._featurize(self.dataset.instances[PLACEHOLDER_FIELD]['0'])
        self.assertEqual(features.shape[0], 1024)

    def test_atom_pair_wrapper(self):
        wrapper = DeepMolDescriptors(n_jobs=2, preset="atompair_fingerprints", kwargs={"nBits": 1024})
        wrapper._fit(self.dataset, PLACEHOLDER_FIELD)
        features = wrapper._featurize(self.dataset.instances[PLACEHOLDER_FIELD]['0'])
        self.assertEqual(features.shape[0], 1024)

    def test_maccs_keys_fingerprints_wrapper(self):
        wrapper = DeepMolDescriptors(n_jobs=2, preset="maccs_keys_fingerprints")
        wrapper._fit(self.dataset, PLACEHOLDER_FIELD)
        features = wrapper._featurize(self.dataset.instances[PLACEHOLDER_FIELD]['0'])
        self.assertEqual(features.shape[0], 167)

    def test_layered_fingerprints_wrapper(self):
        wrapper = DeepMolDescriptors(n_jobs=2, preset="layered_fingerprints")
        wrapper._fit(self.dataset, PLACEHOLDER_FIELD)
        features = wrapper._featurize(self.dataset.instances[PLACEHOLDER_FIELD]['0'])
        self.assertEqual(features.shape[0], 2048)

    def test_rdk_fingerprints_wrapper(self):
        wrapper = DeepMolDescriptors(n_jobs=2, preset="rdk_fingerprints")
        wrapper._fit(self.dataset, PLACEHOLDER_FIELD)
        features = wrapper._featurize(self.dataset.instances[PLACEHOLDER_FIELD]['0'])
        self.assertEqual(features.shape[0], 2048)

    def test_2d_descriptors_wrapper(self):
        wrapper = DeepMolDescriptors(n_jobs=2, preset="2d_descriptors")
        wrapper._fit(self.dataset, PLACEHOLDER_FIELD)
        features = wrapper._featurize(self.dataset.instances[PLACEHOLDER_FIELD]['0'])
        self.assertEqual(features.shape[0], 210)
