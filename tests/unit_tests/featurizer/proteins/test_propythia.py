from plants_sm.featurization.proteins.propythia.propythia import PropythiaWrapper
from ...featurizer.proteins.test_protein_featurizers import TestProteinFeaturizers


class TestPropythiaWrapper(TestProteinFeaturizers):

    def test_all(self):
        wrapper = PropythiaWrapper(preset="all", n_jobs=2)
        wrapper.fit_transform(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.features_dataframe.shape[1], 9597)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_physicochemical(self):
        wrapper = PropythiaWrapper(preset="psycho-chemical", n_jobs=2)
        wrapper.fit_transform(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.features_dataframe.shape[1], 27)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_performance(self):
        wrapper = PropythiaWrapper(preset="performance", n_jobs=2)
        wrapper.fit_transform(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        print(self.dataset.features_dataframe.shape[1])
        self.assertEqual(self.dataset.features_dataframe.shape[1], 8678)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_aac(self):
        wrapper = PropythiaWrapper(preset="aac", n_jobs=2)
        wrapper.fit_transform(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.features_dataframe.shape[1], 8421)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_paac(self):
        wrapper = PropythiaWrapper(preset="paac", n_jobs=2)
        wrapper.fit_transform(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.features_dataframe.shape[1], 71)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_auto_correlation(self):
        wrapper = PropythiaWrapper(preset="auto-correlation", n_jobs=2)
        wrapper.fit_transform(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.features_dataframe.shape[1], 721)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_composition_transition_distribution(self):
        wrapper = PropythiaWrapper(preset="composition-transition-distribution", n_jobs=2)
        wrapper.fit_transform(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.features_dataframe.shape[1], 148)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_seq_order(self):
        wrapper = PropythiaWrapper(preset="seq-order", n_jobs=2)
        wrapper.fit_transform(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.features_dataframe.shape[1], 191)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_modlamp_correlation(self):
        wrapper = PropythiaWrapper(preset="modlamp-correlation", n_jobs=2)
        wrapper.fit_transform(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.features_dataframe.shape[1], 15)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_modlamp_all(self):
        wrapper = PropythiaWrapper(preset="modlamp-all", n_jobs=2)
        wrapper.fit_transform(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.features_dataframe.shape[1], 24)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)


