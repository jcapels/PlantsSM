from unittest import TestCase
from unittest.mock import MagicMock

from pandas import DataFrame

from plants_sm.featurization.proteins.propythia.propythia import PropythiaWrapper
from unit_tests.featurizer.proteins.test_protein_featurizers import TestProteinFeaturizers


class TestPropythiaWrapper(TestProteinFeaturizers):

    def test_all(self):
        wrapper = PropythiaWrapper(preset="all", n_jobs=2)
        wrapper.fit_transform(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.dataframe.shape[1], 9598)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_physicochemical(self):
        wrapper = PropythiaWrapper(preset="psycho-chemical", n_jobs=2)
        wrapper.fit_transform(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.dataframe.shape[1], 28)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_performance(self):
        wrapper = PropythiaWrapper(preset="performance", n_jobs=2)
        wrapper.fit_transform(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.dataframe.shape[1], 8679)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_aac(self):
        wrapper = PropythiaWrapper(preset="aac", n_jobs=2)
        wrapper.fit_transform(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.dataframe.shape[1], 8422)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_paac(self):
        wrapper = PropythiaWrapper(preset="paac", n_jobs=2)
        wrapper.fit_transform(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.dataframe.shape[1], 72)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_auto_correlation(self):
        wrapper = PropythiaWrapper(preset="auto-correlation", n_jobs=2)
        wrapper.fit_transform(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.dataframe.shape[1], 722)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_composition_transition_distribution(self):
        wrapper = PropythiaWrapper(preset="composition-transition-distribution", n_jobs=2)
        wrapper.fit_transform(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.dataframe.shape[1], 149)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_seq_order(self):
        wrapper = PropythiaWrapper(preset="seq-order", n_jobs=2)
        wrapper.fit_transform(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.dataframe.shape[1], 192)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_modlamp_correlation(self):
        wrapper = PropythiaWrapper(preset="modlamp-correlation", n_jobs=2)
        wrapper.fit_transform(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.dataframe.shape[1], 16)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)

    def test_modlamp_all(self):
        wrapper = PropythiaWrapper(preset="modlamp-all", n_jobs=2)
        wrapper.fit_transform(self.dataset)
        self.assertEqual(self.dataset.dataframe.shape[0], 2)
        self.assertEqual(self.dataset.dataframe.shape[1], 25)
        self.assertEqual(wrapper.features_fields, self.dataset.features_fields)


