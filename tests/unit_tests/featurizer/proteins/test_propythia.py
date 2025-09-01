from plants_sm.data_structures.dataset.single_input_dataset import PLACEHOLDER_FIELD
from plants_sm.featurization.proteins.propythia.propythia import PropythiaWrapper
from ...featurizer.proteins.test_protein_featurizers import TestProteinFeaturizers


class TestPropythiaWrapper(TestProteinFeaturizers):

    def test_propythia_wrapper(self):
        wrapper = PropythiaWrapper(preset="all", n_jobs=2)
        wrapper._fit(self.dataset, PLACEHOLDER_FIELD)
        features = wrapper._featurize(list(self.dataset.get_instances().values())[0])
        self.assertEqual(features.shape[0], 9595)

    def test_all(self):
        wrapper = PropythiaWrapper(preset="all", n_jobs=2)
        wrapper.fit_transform(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.X().shape[1], 9595)
        self.assertEqual(wrapper.features_names, self.dataset.features_fields[PLACEHOLDER_FIELD])

    def test_physicochemical(self):
        wrapper = PropythiaWrapper(preset="physicochemical", n_jobs=2)
        wrapper.fit_transform(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.X().shape[1], 25)
        self.assertEqual(wrapper.features_names, self.dataset.features_fields[PLACEHOLDER_FIELD])

    def test_performance(self):
        wrapper = PropythiaWrapper(preset="performance", n_jobs=2)
        wrapper.fit_transform(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.X().shape[1], 8676)
        self.assertEqual(wrapper.features_names, self.dataset.features_fields[PLACEHOLDER_FIELD])

    def test_aac(self):
        wrapper = PropythiaWrapper(preset="aac", n_jobs=2)
        wrapper.fit_transform(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.X().shape[1], 8420)
        self.assertEqual(wrapper.features_names, self.dataset.features_fields[PLACEHOLDER_FIELD])

    def test_paac(self):
        wrapper = PropythiaWrapper(preset="paac", n_jobs=2)
        wrapper.fit_transform(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.X().shape[1], 70)
        self.assertEqual(wrapper.features_names, self.dataset.features_fields[PLACEHOLDER_FIELD])

    def test_auto_correlation(self):
        wrapper = PropythiaWrapper(preset="auto-correlation", n_jobs=2)
        wrapper.fit_transform(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.X().shape[1], 720)
        self.assertEqual(wrapper.features_names, self.dataset.features_fields[PLACEHOLDER_FIELD])

    def test_composition_transition_distribution(self):
        wrapper = PropythiaWrapper(preset="composition-transition-distribution", n_jobs=2)
        wrapper.fit_transform(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.X().shape[1], 147)
        self.assertEqual(wrapper.features_names, self.dataset.features_fields[PLACEHOLDER_FIELD])

    def test_seq_order(self):
        wrapper = PropythiaWrapper(preset="seq-order", n_jobs=2)
        wrapper.fit_transform(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.X().shape[1], 190)
        self.assertEqual(wrapper.features_names, self.dataset.features_fields[PLACEHOLDER_FIELD])

    def test_modlamp_correlation(self):
        wrapper = PropythiaWrapper(preset="modlamp-correlation", n_jobs=2)
        wrapper.fit_transform(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.X().shape[1], 14)
        self.assertEqual(wrapper.features_names, self.dataset.features_fields[PLACEHOLDER_FIELD])

    def test_modlamp_all(self):
        wrapper = PropythiaWrapper(preset="modlamp-all", n_jobs=2)
        wrapper.fit_transform(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.X().shape[1], 23)
        self.assertEqual(wrapper.features_names, self.dataset.features_fields[PLACEHOLDER_FIELD])
