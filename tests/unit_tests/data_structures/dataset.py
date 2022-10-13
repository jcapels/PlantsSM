from unittest import TestCase

import numpy as np
import pandas as pd

from plants_sm.data_structures.dataset import PandasDataset


class TestDataset(TestCase):

    def test_pandas_dataset(self):
        """
        Test the PandasDataset class.
        """
        dataframe = pd.DataFrame(columns=['a', 'b', 'c'])

        dataframe.loc[0] = ["representation2", 2, 3]
        dataframe.loc[1] = ["representation1", 5, 6]
        dataframe.loc[2] = ["representation", 8, 9]

        dataset = PandasDataset(dataframe, representation_field='a', labels_field='c', features_fields='b')
        self.assertTrue(dataset.representation_field == 'a')
        self.assertTrue(dataset.features_fields == ['b'])
        self.assertTrue(dataset.labels_names == ['c'])
        self.assertTrue(all(representation in dataset.instances for representation in ['representation2',
                                                                                       'representation1',
                                                                                       'representation']))
        self.assertTrue(all(feature in dataset.features for feature in [2, 5, 8]))
        self.assertTrue(all(label in dataset.labels for label in [3, 6, 9]))

        dataset = PandasDataset(dataframe, representation_field='a', labels_field='c', features_fields=[1])
        self.assertTrue(dataset.features_fields == ["b"])
        self.assertEqual(dataset.features.shape[1], 1)

    def test_drop_nan(self):
        dataframe = pd.DataFrame(columns=['a', 'b', 'c'])

        dataframe.loc[0] = ["representation2", 2, np.nan]
        dataframe.loc[1] = ["representation1", 5, 6]
        dataframe.loc[2] = ["representation", 8, 9]

        dataset = PandasDataset(dataframe, representation_field='a', labels_field='c', features_fields='b')
        dataset.drop_nan()
        self.assertEqual(2, len(dataset.instances))

    def test_drop_nan_columns(self):
        dataframe = pd.DataFrame(columns=['a', 'b', 'c'])

        dataframe.loc[0] = [np.nan, 2, np.nan]
        dataframe.loc[1] = ["representation1", 5, 6]
        dataframe.loc[2] = ["representation", 8, 9]

        dataset = PandasDataset(dataframe, representation_field='a', labels_field='c', features_fields='b')
        with self.assertRaises(ValueError):
            dataset.drop_nan(axis=1)

        dataframe = pd.DataFrame(columns=['a', 'b', 'c'])

        dataframe.loc[0] = ["representation2", 2, np.nan]
        dataframe.loc[1] = ["representation1", 5, 6]
        dataframe.loc[2] = ["representation", 8, 9]

        dataset = PandasDataset(dataframe, representation_field='a', labels_field='c', features_fields='b')

        with self.assertRaises(ValueError):
            dataset.drop_nan(axis=1)

        dataframe = pd.DataFrame(columns=['a', 'b', 'c'])

        dataframe.loc[0] = ["representation2", np.nan, 3]
        dataframe.loc[1] = ["representation1", 5, 6]
        dataframe.loc[2] = ["representation", 8, 9]

        dataset = PandasDataset(dataframe, representation_field='a', labels_field='c', features_fields='b')

        dataset.drop_nan(axis=1)

        self.assertEqual(3, len(dataset.instances))
        features = dataset.features
        self.assertEqual(0, features.size)

    def test_slice_as_features_field(self):
        dataframe = pd.DataFrame(columns=['a', 'b', 'c'])

        dataframe.loc[0] = ["representation2", 2, 3]
        dataframe.loc[1] = ["representation1", 5, 6]
        dataframe.loc[2] = ["representation", 8, 9]

        dataset = PandasDataset(dataframe, representation_field='a', features_fields=slice(1, None))
        self.assertTrue(dataset.features.shape[1] == 2)

